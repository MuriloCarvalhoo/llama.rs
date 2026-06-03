use crate::error::OracleError;

/// Extrai os IDs da saída `--ids` do llama-tokenize (formato `[1, 2, 3]`,
/// possivelmente cercado de logs em outras linhas).
pub fn parse_token_ids(output: &str) -> Result<Vec<i64>, OracleError> {
    let err = || OracleError::Parse(output.to_owned());
    let start = output.find('[').ok_or_else(err)?;
    let end = output.rfind(']').ok_or_else(err)?;
    let inner = output.get(start.saturating_add(1)..end).ok_or_else(err)?;
    let inner = inner.trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|tok| {
            tok.trim()
                .parse::<i64>()
                .map_err(|_| OracleError::Parse(tok.to_owned()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bracketed_id_list() {
        let ids = parse_token_ids("[1, 15043, 3186]").unwrap();
        assert_eq!(ids, vec![1, 15043, 3186]);
    }

    #[test]
    fn parses_ids_with_surrounding_log_noise() {
        let out = "load: vocab loaded\n[1, 2, 3]\n";
        assert_eq!(parse_token_ids(out).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn parses_empty_list() {
        assert_eq!(parse_token_ids("[]").unwrap(), Vec::<i64>::new());
    }

    #[test]
    fn rejects_output_without_brackets() {
        assert!(matches!(
            parse_token_ids("error: model not found"),
            Err(OracleError::Parse(_))
        ));
    }

    #[test]
    fn rejects_non_numeric_entries() {
        assert!(matches!(
            parse_token_ids("[1, x, 3]"),
            Err(OracleError::Parse(_))
        ));
    }
}
