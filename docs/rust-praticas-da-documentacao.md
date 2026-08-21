# Rust: memória, velocidade e más práticas segundo a documentação oficial

**Data:** 2026-08-21 · **Toolchain:** rustc 1.98.0 stable · **Validação:** `docs/validacao-rust-praticas/`

Este documento é irmão de `rust-memoria-e-desempenho.md` (pesquisa aplicada ao llama-rs:
Vulkan, mmap, NUMA, alocadores). Aqui o escopo é o inverso: o que a **documentação oficial
do Rust** afirma sobre economizar memória, ser mais rápido e o que evitar — e o que dessas
afirmações se confirma **medindo nesta máquina** com dezenas a centenas de milhões de elementos.

**Fontes** (nada aqui foi inventado — cada item cita a sua):

- **[Perf Book]** The Rust Performance Book — <https://nnethercote.github.io/perf-book/>
- **[std]** Documentação da biblioteca padrão — <https://doc.rust-lang.org/std/>
- **[Cargo]** Cargo Book, referência de profiles — <https://doc.rust-lang.org/cargo/reference/profiles.html>
- **[Clippy]** Índice de lints — <https://rust-lang.github.io/rust-clippy/master/index.html>

**Legenda:** ✅ = confirmado por medição local (§5) · 📖 = afirmação da fonte, não medida aqui
· ⚠️ = a fonte manda medir antes de adotar.

---

## 1. Economizar memória

### 1.1 Pré-alocar em vez de crescer

| O quê | Fonte |
|---|---|
| ✅ `Vec::with_capacity(n)` quando o tamanho é conhecido. Crescer por `push` faz alocações sucessivas (a doc exemplifica: chegar a 20 elementos custa 4 alocações — capacidades 4, 8, 16, 32) e **cada realocação copia tudo** | [Perf Book] Heap Allocations; [std] Vec "Capacity and reallocation": "é recomendado usar `Vec::with_capacity` sempre que possível" |
| 📖 `reserve`/`reserve_exact` para o mesmo fim em vetor já existente; `reserve` "não faz nada se a capacidade já é suficiente" | [std] Vec; [Perf Book] Heap Allocations |
| ✅ `HashMap::with_capacity` / `HashSet::with_capacity` — mesma lógica | [Perf Book] Heap Allocations; [std] HashMap |
| ✅ `String::with_capacity` ao anexar muitos dados — o exemplo da doc mostra capacidades 0→8→16→32 com `new()` contra capacidade constante com `with_capacity` | [std] String |

Nuance medida aqui (§5, `vec_crescimento` e `encolher`): para **um** vetor gigante no
Linux/glibc a diferença é modesta (~9% no tempo, pico de RSS idêntico — realocações grandes
usam `mremap`, sem copiar, e capacidade não tocada não vira RSS). O caso que a doc exemplifica
— 4 alocações para chegar a 20 elementos — é o de **muitos vetores pequenos/médios**, onde o
custo é por alocação, não por cópia.

### 1.2 Devolver o que sobrou

- ✅ **`Vec` nunca encolhe sozinho**: "`Vec` nunca vai encolher automaticamente, mesmo se
  completamente vazio" ([std] Vec "Guarantees"). Quem quer devolver memória chama
  `shrink_to_fit` — que "pode manter algum excesso de capacidade" e pode realocar.
- ✅ `Vec::into_boxed_slice` descarta o excesso como `shrink_to_fit` e o handle cai de
  **3 palavras (ptr, len, cap) para 2 (ptr, len)** — medido: `Vec<u64>` = 24 bytes,
  `Box<[u64]>` = 16 ([std] Vec; [Perf Book] Type Sizes).

### 1.3 Não clonar o que pode ser emprestado

- ✅ Remover `clone` desnecessário — "código Rust acaba contendo `clone` desnecessários por
  erro do programador ou por mudanças posteriores" ([Perf Book] Heap Allocations). O lint
  `redundant_clone` existe mas está no grupo **nursery** (análise conservadora, desligado por
  padrão) — [Clippy].
- 📖 `clone_from` em vez de `clone` quando há destino já alocado — reutiliza a alocação
  ([Perf Book] Heap Allocations).
- 📖 `Cow<'_, str>`/`Cow<'_, [T]>`: "clona os dados preguiçosamente quando mutação ou posse
  for necessária" — caminho sem mutação não aloca ([std] Cow).
- 📖 `Rc::make_mut`/`Arc::make_mut`: clone-on-write — só clona se o refcount > 1
  ([Perf Book] Standard Library Types).
- 📖 Evitar `Rc`/`Arc` para valores raramente compartilhados — põem no heap o que talvez nem
  precisasse de heap ([Perf Book] Heap Allocations).
- 📖 Reusar coleções: receber `&mut Vec` para preencher; ou "workhorse collection" — declarar
  o `Vec` fora do laço e `clear()` a cada iteração ([Perf Book] Heap Allocations).
- 📖 Ao ler linhas: `BufRead::read_line` com `String` reutilizada em vez de `lines()` — cai de
  uma alocação por linha para "no máximo um punhado" ([Perf Book] Heap Allocations).

### 1.4 Encolher os tipos

- ✅ **Niche optimization garantida**: `Option<T>` tem o mesmo tamanho que `T` para `Box<U>`,
  `&U`, `&mut U`, `fn`, `NonZero*`, `NonNull<U>` ([std] option "Representation"). Medido:
  `Option<Box<u64>>` = 8 bytes, `Option<&u64>` = 8, `Option<NonZeroU64>` = 8.
- ✅ Variante grande de enum em `Box`: o enum inteiro tem o tamanho da maior variante. Medido:
  enum com variante `[u8; 256]` inline = 257 bytes; com `Box<[u8; 256]>` = 16
  ([Perf Book] Type Sizes; [Clippy] `large_enum_variant`, com o aviso da doc: **medir** — se a
  variante pequena for rara, o box é contraproducente).
- 📖 O mesmo para erros: `Result` tem no mínimo o tamanho do `Err`, e esse custo se propaga
  pela pilha via `?` ([Clippy] `result_large_err`).
- 📖 Índices `u32`/`u16`/`u8` em vez de `usize` quando o alcance permite, convertendo no uso
  ([Perf Book] Type Sizes).
- 📖 **Não** reordenar campos manualmente para economizar padding — o compilador já otimiza a
  ordem (exceto `#[repr(C)]`) ([Perf Book] Type Sizes).
- 📖 `String` → `Box<str>` segue a mesma lógica do `Vec` → `Box<[T]>` (24 → 16 bytes, medido
  em §5 `tamanhos`).

### 1.5 Escolher a coleção certa

Da página [std] `std::collections` — "When Should You Use Which Collection?":

| Coleção | Quando (segundo a doc) | Custos documentados |
|---|---|---|
| `Vec` | sequência, append no fim, pilha, "array no heap" — **o padrão** | get O(1), insert/remove O(n−i) |
| `VecDeque` | fila, inserção eficiente nas duas pontas | get O(1), insert/remove O(min(i, n−i)) |
| `LinkedList` | só se você está "*absolutamente* certo de que *realmente*, *de verdade*, quer uma lista duplamente encadeada" | get/insert/remove O(min(i, n−i)) |
| `HashMap` | mapa/cache sem funcionalidade extra | get/insert/remove O(1) esperado |
| `BTreeMap` | mapa **ordenado** pelas chaves, ranges, menor/maior chave | tudo O(log n) |
| `BinaryHeap` | fila de prioridade | push O(1) amortizado, pop O(log n) |

E a regra de desempate da própria doc: "onde há empate, `Vec` é geralmente mais rápido que
`VecDeque`, e `VecDeque` é geralmente mais rápido que `LinkedList`". ✅ HashMap vs BTreeMap
medido em §5 (`mapa_insercao`): tempo **e** memória.

### 1.6 Crates externos que o Perf Book cita para memória

Todos ⚠️ (o livro manda medir; nenhum é std): `smallvec` (N elementos inline; "reduz a taxa
de alocação de forma confiável, mas **não garante** melhora de desempenho"), `arrayvec`
(comprimento máximo conhecido, sem fallback de heap — "um pouco mais rápido que SmallVec"),
`thin-vec` (handle de 1 palavra), `smartstring` (strings < 24 bytes sem heap). Fonte:
[Perf Book] Heap Allocations e Type Sizes.

---

## 2. Ser mais rápido

### 2.1 Build: o que ligar no `Cargo.toml`

Defaults exatos do perfil `release` ([Cargo] profiles): `opt-level = 3`, `lto = false`,
`codegen-units = 16`, `panic = 'unwind'`, `debug = false`, `incremental = false`.

| Flag | O que a fonte afirma | Status |
|---|---|---|
| build de release | "speedups de 10-100x sobre dev builds são comuns" ([Perf Book] Build Configuration) | ✅ medido em §5 (`soma_matmul`) |
| `lto = "thin"`/`"fat"` | "pode melhorar a velocidade em 10-20% ou mais" ([Perf Book]); `"thin"` roda em "substancialmente menos tempo com ganhos parecidos" ao fat ([Cargo]) | já ligado neste repo (`lto = "thin"`, com medições próprias em `rust-memoria-e-desempenho.md` §2.6) |
| `codegen-units = 1` | mais unidades compilam em paralelo "mas podem produzir código mais lento" ([Cargo]); uma única unidade permite código melhor ([Perf Book]) | já ligado neste repo |
| `-C target-cpu=native` | "pode melhorar a velocidade, especialmente se o compilador encontra oportunidades de vetorização" ([Perf Book]) | já ligado neste repo (`.cargo/config.toml`) |
| `opt-level` | aviso da doc: nível `3` **pode ser mais lento** que `2`, e `"s"`/`"z"` não são necessariamente menores; reavaliar a cada versão do rustc ([Cargo]) | 📖 |
| `panic = "abort"` | "pode reduzir o binário e aumentar a velocidade levemente" ([Perf Book]); testes/benches **ignoram** a opção ([Cargo]) | fora neste repo: quebra `#[should_panic]` (nota no `Cargo.toml` raiz) |
| PGO (`cargo-pgo`) | "10% ou mais" ([Perf Book]) | ⚠️ ver a leitura honesta em `rust-memoria-e-desempenho.md` §2.6: PGO recupera desempenho deixado na mesa, não cria |
| alocador jemalloc/mimalloc | "podem trazer grandes melhorias" ([Perf Book] — crates externos) | ⚠️ no padrão deste projeto (alocações grandes e longevas) o efeito medido publicado é ~zero — `rust-memoria-e-desempenho.md` §1.5 |

### 2.2 Iteradores

- ✅ **Não** fazer `collect` intermediário quando a coleção só será iterada de novo — `collect`
  "tipicamente requer uma alocação"; preferir a cadeia fundida ou retornar
  `impl Iterator<Item = T>` ([Perf Book] Iterators). Medido: `collect_intermediario`.
- 📖 `extend` para anexar um iterador a coleção existente, em vez de `collect` + `append`
  ([Perf Book] Iterators).
- 📖 Implementar `size_hint`/`ExactSizeIterator::len` em iteradores próprios — `collect` e
  `extend` alocam menos ([Perf Book] Iterators).
- 📖 `filter_map` em vez de `filter` + `map`; `chunks_exact` em vez de `chunks`; `copied()` ao
  iterar tipos pequenos ("LLVM pode gerar código melhor"); evitar `chain` em caminho quente
  ([Perf Book] Iterators).

### 2.3 Bounds checks

Do [Perf Book] Bounds Checks, em ordem de preferência:

1. Substituir indexação em laço por **iteração** (`iter()`, `zip`) — o compilador elimina as
   verificações. **Medido aqui e a diferença foi ~zero** (§5, `produto_escalar`): num laço
   limitado por banda de memória (800 MB lidos a ~9,5 GB/s), o custo do bounds check some. A
   recomendação segue válida para laços limitados por computação — mas este resultado é o
   lembrete empírico de que o próprio Perf Book manda medir.
2. 📖 Criar um slice local antes do laço e indexar o slice — o compilador rastreia melhor o
   comprimento.
3. 📖 `assert!` sobre os ranges dos índices antes do laço.
4. `get_unchecked` é o último recurso e é `unsafe` — **vedado neste repo**
   (`unsafe_code = "deny"`).

### 2.4 Ordenação

- ✅ `sort_unstable` "é tipicamente mais rápido que a ordenação estável e **não aloca memória
  auxiliar**"; `sort` (estável, driftsort) aloca até `len/2`. A exceção documentada: slices
  parcialmente ordenadas podem favorecer `sort` ([std] slice). Medido: tempo e pico de RSS em
  `ordenar`.

### 2.5 Hashing

- O hasher padrão do `HashMap` é **SipHash 1-3**: "muito competitivo para chaves de tamanho
  médio", mas superado "para chaves pequenas, como inteiros, bem como chaves grandes, como
  strings longas" — e as alternativas "tipicamente *não* protegem contra HashDoS" ([std] HashMap).
- 📖 `rustc-hash` (`FxHashMap`): no rustc, trocar deu speedups de **até 6%**; `ahash`: a
  tentativa no rustc deu **lentidão de 1-4%** — ou seja, trocar hasher sem medir é aposta
  ([Perf Book] Hashing). ⚠️ "Só trocar se o profiling mostrar que hashing está quente."
- ✅ `entry` API em vez de `contains_key`/`get_mut` + `insert` — uma busca em vez de duas
  ([Clippy] `map_entry`). Medido: `mapa_entry`.

### 2.6 I/O

- ✅ `BufWriter`/`BufReader` para chamadas **pequenas e repetidas** — cada `read`/`write`
  direto num `File`/`TcpStream` é uma syscall; o buffer agrupa ([std] BufReader/BufWriter;
  [Perf Book] I/O). Medido: `io_escrita`, `io_leitura`.
- ✅ A mesma doc delimita: **não** ajudam "ao escrever quantidades muito grandes de uma vez, ou
  ao escrever apenas uma ou poucas vezes", nem para destinos em memória (`Vec<u8>`). Medido:
  `write_all` único é a variante mais rápida de todas.
- 📖 `flush()` explícito no `BufWriter` — erros no flush do drop são **ignorados** ([std]
  BufWriter; motivo declarado é corretude, não velocidade).
- 📖 Muitos `println!` travam stdout a cada chamada — fazer lock manual e combiná-lo com
  buffering ([Perf Book] I/O).
- 📖 Ler bytes crus (`read_until`) quando não precisa de `String` — validação UTF-8 tem custo
  "pequeno, mas não nulo" ([Perf Book] I/O).

### 2.7 Inlining e layout de código

Do [Perf Book] Inlining — tudo ⚠️ ("às vezes não tem efeito; às vezes **deixa mais lento**;
medir antes e depois"):

- `#[inline]` em funções muito pequenas ou com um único callsite.
- Inlining **não é transitivo**: em cadeia quente `f → g`, anotar as duas.
- Padrão split para função com callsites quentes e frios: versão `#[inline(always)]` no quente,
  `#[inline(never)]` nos frios.
- `#[cold]` em funções raramente executadas melhora o código do caminho quente.

### 2.8 Dicas gerais com respaldo

Do [Perf Book] General Tips: otimizar **só o código quente**; mudança de algoritmo/estrutura de
dados vale mais que micro-otimização; fast paths otimistas para os casos comuns; tratamento
especial de coleções com 0/1/2 elementos "é frequentemente uma vitória"; cache pequeno para
lookups com localidade; acesso à memória sequencial > aleatório (minimizar cache misses);
computar lazy (`ok_or_else` e demais variantes `_else` só avaliam quando precisa —
[Perf Book] Standard Library Types).

Operações O(1) que substituem O(n): `swap_remove` em vez de `remove` quando a ordem não
importa; `retain` para remoção em massa ([Perf Book] Standard Library Types).

Paralelismo: o capítulo do [Perf Book] é um apontador para `rayon`/`crossbeam` (crates
externos), sem números. Para o custo real do rayon **neste projeto**, ver
`rust-memoria-e-desempenho.md` §2.2.

---

## 3. Más práticas: o catálogo oficial

### 3.1 A pior de todas (medida): concatenação quadrática de String

A doc de `String` explica por que `+` consome o lado esquerdo e reutiliza o buffer: alocar
uma nova `String` e copiar tudo a cada passo "levaria a tempo de execução **O(n²)**" ([std]
String, impl `Add<&str>`). É exatamente o que `s = format!("{s}{x}")` num laço faz.
✅ Medido em §5 (`string_concat`): a diferença contra `push_str`/`write!` é de **três ordens
de grandeza** com apenas 100 mil itens.

### 3.2 Os 38 lints do grupo `perf` do Clippy

Cada lint é uma má prática documentada oficialmente, com a correção na própria doc
(<https://rust-lang.github.io/rust-clippy/master/index.html#nome_do_lint>). O grupo `perf` é
warn por padrão — `cargo clippy` já acusa tudo isto:

| Lint | Má prática → correção |
|---|---|
| `box_collection` | `Box<Vec<T>>` etc. — coleção já é heap → usar a coleção direta |
| `boxed_local` | `Box<T>` onde `T` bastaria → sem box |
| `cloned_ref_to_slice_refs` | `&[x.clone()]` → `slice::from_ref(&x)` |
| `cmp_owned` | criar valor owned só para comparar → comparar pela referência |
| `collapsible_str_replace` | `replace().replace()` varre a string 2× → uma chamada |
| `double_ended_iterator_last` | `last()` consome o iterador inteiro → `next_back()` |
| `drain_collect` | `drain(..).collect()` → `mem::take` (evita alocação) |
| `expect_fun_call` | `expect(&format!(...))` avalia sempre → `unwrap_or_else` |
| `extend_with_drain` | `extend(v.drain(..))` → `append` |
| `format_in_format_args` | `format!` dentro de outra macro de formato → inline no formato externo |
| `iter_overeager_cloned` | `cloned()` antes de `filter`/`take` clona o que será descartado → adiar o `cloned` |
| `large_const_arrays` | array `const` grande é inlinado a cada uso → `static` |
| `large_enum_variant` | variante muito maior que as outras → `Box` nela (e **medir**) |
| `manual_clear` | `truncate(0)` → `clear()` |
| `manual_contains` | `iter().any(...)` em slice → `contains()` |
| `manual_ignore_case_cmp` | `to_ascii_lowercase()` dos dois lados aloca → `eq_ignore_ascii_case` |
| `manual_memcpy` | laço copiando entre slices → `copy_from_slice` (memcpy) |
| `manual_retain` | `filter().collect()` reatribuído → `retain()` |
| `manual_str_repeat` | repetição manual de string → `str::repeat` |
| `manual_try_fold` | `fold` com `Try` não faz short-circuit → `try_fold` |
| `map_entry` | `contains_key` + `insert` (2 buscas) → `entry` API ✅ |
| `missing_const_for_thread_local` | `thread_local!` sem `const` → `const { ... }` evita init preguiçosa |
| `missing_spin_loop` | spin loop de corpo vazio → `hint::spin_loop()` / lock de verdade |
| `readonly_write_lock` | `RwLock::write` só para ler bloqueia todo mundo → `read` |
| `redundant_allocation` | `Rc<Box<T>>`, `Box<Box<T>>`… → remover a camada extra |
| `redundant_iter_cloned` | `cloned()` onde o original serviria → iterar o original |
| `regex_creation_in_loops` | compilar regex dentro de laço → compilar uma vez fora |
| `replace_box` | `*b = Box::new(v)` aloca de novo → atribuir ao conteúdo (`*b = v`) |
| `result_large_err` | `Err` gigante infla todo `Result` na pilha → box no erro |
| `sliced_string_as_bytes` | `s[a..b].as_bytes()` valida UTF-8 à toa → `&s.as_bytes()[a..b]` |
| `slow_vector_initialization` | `Vec::new()` + `resize(n, 0)` → `vec![0; n]` (usa `alloc_zeroed`) ✅ |
| `to_string_in_format_args` | `to_string()` dentro de macro de formato → passar o valor direto |
| `unbuffered_bytes` | `Read::bytes` sem `BufRead` = 1 `read` por byte → `BufReader` ✅ |
| `unnecessary_to_owned` | `to_owned`/`to_vec` desnecessário → usar o emprestado |
| `useless_borrows_in_formatting` | `&x` extra em macro de formato vira `&&T` (doc: ~6% por chamada) → remover o `&` |
| `useless_vec` | `vec![..]` onde array `[..]` basta → array |
| `vec_init_then_push` | `Vec::new()` + sequência de `push` → macro `vec![]` |
| `waker_clone_wake` | `waker.clone().wake()` → `wake_by_ref()` |

Notas de status ([Clippy], verificado no master em 2026-08-21): `redundant_clone` está em
**nursery** (conservador, allow por padrão); `single_char_pattern` e `inefficient_to_string`
estão em **pedantic** — o primeiro com benchmarks declarados inconclusivos pela própria doc, o
segundo relevante só antes do Rust 1.82. `manual_clear` e `useless_borrows_in_formatting`
entraram em 1.97; `replace_box` e `redundant_iter_cloned` em 1.92.

### 3.3 Más práticas fora do Clippy, com fonte

- ✅ Clonar dentro de laço quente o que podia ser emprestado ([Perf Book] Heap Allocations).
  Medido: `clone_em_laco` — e o custo não é só o memcpy: é 10 milhões de pares malloc/free.
- 📖 `format!` quando um literal ou `write!` no buffer basta — toda chamada aloca uma `String`
  ([Perf Book] Heap Allocations).
- 📖 Confiar em `LinkedList` por hábito de outras linguagens — a doc da std praticamente pede
  desculpas por ela existir (§1.5).
- 📖 Trocar de hasher/alocador/`parking_lot` **sem medir** — as três páginas correspondentes do
  Perf Book mandam medir antes; o caso ahash-no-rustc (§2.5) mostra troca "óbvia" que piorou.
- Más práticas específicas deste projeto (async para GPU, `Arc<[u8]>::from(vec)` copiando
  15 GB, `queue_wait_idle` por submissão): já documentadas em `rust-memoria-e-desempenho.md`.

---

## 4. Aplicação neste repositório

- `[profile.release]` com `lto = "thin"` + `codegen-units = 1` já segue §2.1; `panic = "abort"`
  descartado com razão documentada no próprio `Cargo.toml`.
- `target-cpu=native` já em `.cargo/config.toml`.
- O workspace nega `unsafe_code` — portanto a escada de bounds checks (§2.3) para na etapa 3
  (assertions); `get_unchecked` não é opção aqui.
- `indexing_slicing = "warn"` no workspace já empurra para iteradores/`get` — coerente com §2.3.
- O grupo `perf` do Clippy é warn por padrão; nada a configurar para ganhar §3.2 inteira.

## 5. Resultados medidos nesta máquina

**Ambiente:** 2× Xeon E5-2680 v4 (28 núcleos, 2 nós NUMA) · kernel 7.2.0-rc7 CachyOS · glibc ·
rustc 1.98.0 · release com `lto="thin"`, `codegen-units=1`, `target-cpu=native` · mediana de
3 repetições, um processo por variante. Dados brutos: `docs/validacao-rust-praticas/resultados.csv`.

| Experimento (n) | Variante | Mediana | Pico RSS |
|---|---|---:|---:|
| `vec_crescimento` (100M u64) | `push` sem capacidade | 390 ms | 765 MB |
| | `with_capacity` | 357 ms | 765 MB |
| `string_concat` (100 mil itens) | `s = format!("{s}{i},")` | **11 714 ms** | 3 MB |
| | `push_str` | 3,9 ms | 3 MB |
| | `write!` + capacidade | 5,4 ms | 2 MB |
| `produto_escalar` (100M u32) | indexado `a[i]*b[i]` | 84 ms | 765 MB |
| | `zip` de iteradores | 82 ms | 765 MB |
| `ordenar` (50M u64 aleatórios) | `sort` (estável) | 3 679 ms | **574 MB** |
| | `sort_unstable` | 2 171 ms | 383 MB |
| `mapa_insercao` (10M chaves u64) | `HashMap::new` | 1 653 ms | 486 MB |
| | `HashMap::with_capacity` | 1 304 ms | 350 MB |
| | `BTreeMap` | 7 198 ms | 355 MB |
| `clone_em_laco` (10M clones de String ~32 B) | `clone` | 144 ms | 138 MB |
| | empréstimo | 26,5 ms | 139 MB |
| `io_escrita` (2M writes de 1 byte) | `File` direto | 1 429 ms | 2 MB |
| | `BufWriter` | 6,9 ms | 2 MB |
| | `write_all` único | 2,6 ms | 4 MB |
| `io_leitura` (2M reads de 1 byte) | `File` direto | 1 001 ms | 4 MB |
| | `BufReader` | 6,9 ms | 4 MB |
| `collect_intermediario` (100M u32) | `collect` + iterar de novo | 386 ms | **1 146 MB** |
| | cadeia fundida | 57 ms | 383 MB |
| `vec_inicializacao` (500M u8, criar+ler) | `vec![0; n]` | 79 ms | **2 MB** |
| | `with_capacity` + `push(0)` | 816 ms | 479 MB |
| | `Vec::new` + `resize` | 237 ms | 479 MB |
| `mapa_entry` (10M ops, 5M chaves) | `get_mut`/`insert` (2 buscas) | 1 298 ms | 282 MB |
| | `entry` API | 1 210 ms | 282 MB |
| `soma_matmul` (100M + matmul 384³) | perfil **dev** | 4 178 ms | 385 MB |
| | perfil **release** | 62 ms | 385 MB |

### Vereditos, experimento a experimento

- **Concatenação quadrática de String — a pior má prática medida.** `format!("{s}{i},")` num
  laço: **3 000× mais lento** que `push_str` já com 100 mil itens (e piora quadraticamente).
  Confirma o O(n²) descrito na doc de `String`. As duas variantes lineares empatam — a
  capacidade prévia não fez diferença mensurável nesta escala de bytes (~589 KB).
- **dev vs release: 68×.** Dentro da faixa "10-100x" do Perf Book. Nunca medir nada em dev.
- **I/O sem buffer: 207× (escrita) e 145× (leitura).** O caso mais extremo depois da string
  quadrática. E `write_all` único ainda é 2,7× mais rápido que `BufWriter` — como a doc do
  `BufWriter` afirma, buffer não ajuda quando se escreve tudo de uma vez.
- **`collect` intermediário: 6,8× mais lento e +763 MB de pico** (o `Vec<u64>` intermediário
  de 800 MB aparece inteiro no RSS). A confirmação mais forte do capítulo Iterators.
- **`vec![0; n]`: 3× mais rápido que `resize`, 10× que `push` — e RSS de 2 MB** contra 479 MB:
  o `alloc_zeroed` mapeia a página-zero do kernel, e ler zeros nunca materializa memória
  física. Confirma `slow_vector_initialization` com um bônus de memória que o lint nem promete.
- **`sort_unstable`: 1,7× mais rápido e −191 MB de pico** — exatamente o "aloca até len/2"
  documentado (50M × 8 B / 2 = 200 MB de auxiliar do `sort` estável).
- **`HashMap::with_capacity`: 21% mais rápido e −28% de pico** (sem os rehashes de
  crescimento). **`BTreeMap`: 4,4× mais lento** que `HashMap` para inserção aleatória —
  o O(log n) vs O(1) da tabela da std, em números.
- **`clone` em laço quente: 5,4×** — 10 milhões de pares malloc/memcpy/free contra leitura
  emprestada.
- **`entry` API: ~7% mais rápido** (venceu nas 3 repetições, ganho modesto). Vale pelo estilo;
  não espere milagre.
- **`with_capacity` num único vetor gigante: ~9%, pico idêntico.** A surpresa honesta: glibc
  usa `mremap` para realocações grandes (sem cópia física) e capacidade não tocada não vira
  RSS. O conselho da doc rende mais em muitos vetores pequenos/médios.
- **Bounds checks: diferença ~zero neste laço** — 800 MB lidos a ~9,5 GB/s; a banda de memória
  domina e esconde o custo da verificação. Ver §2.3.
- **`shrink_to_fit`** (100M u64 crescidos por push): capacidade 1 024 → 762 MB em 0,1 ms
  (`mremap`, sem cópia), RSS inalterado — as páginas excedentes nunca tinham sido tocadas.
  Devolve espaço de endereçamento e capacidade; só devolve RSS se o excesso foi escrito.
- **Tamanhos de tipo** (asserções, não tempo): `Option<Box<u64>>` = 8 B, `Option<&u64>` = 8 B,
  `Option<NonZeroU64>` = 8 B (niche optimization garantida); `Vec<u64>` 24 B vs `Box<[u64]>`
  16 B; `String` 24 B vs `Box<str>` 16 B; enum com variante `[u8; 256]`: 257 B inline vs 16 B
  com `Box`.

## 6. Como rodar a validação

```sh
bash docs/validacao-rust-praticas/rodar.sh          # 3 repetições, ~4 min
REPS=5 bash docs/validacao-rust-praticas/rodar.sh   # mais repetições
```

Saída bruta em `docs/validacao-rust-praticas/resultados.csv`
(`resultado,experimento,variante,n,ms,pico_mb` + linhas `info,...`); medianas no stderr ao
final. Cada variante roda num **processo separado** porque `VmHWM` (pico de RSS em
`/proc/self/status`) é monotônico por processo. Um experimento aceita `n` custom:
`target/release/validacao-rust-praticas ordenar sort 10000000`.

## 7. Limites deste documento

- O que está 📖 é afirmação de fonte confiável, **não** verificada nesta máquina; o que está ✅
  foi medido aqui (§5) — em outra CPU/kernel/alocador os números mudam, o sinal raramente.
- O Perf Book afirma vários efeitos sem número ("pode ser mais rápido"); esses itens estão
  reproduzidos com a mesma modéstia da fonte.
- Benchmarks locais medem operações isoladas em dados sintéticos grandes; efeitos de cache/ICache
  em código real podem divergir — é a razão pela qual o Perf Book manda perfilar o programa
  real antes de otimizar (§2.8).
