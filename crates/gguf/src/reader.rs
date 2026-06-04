//! Cursor com bounds-check sobre um slice `&[u8]`. Todo método retorna
//! `Result` em vez de panicar — GGUF é entrada não-confiável.

use crate::error::GgufError;

pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    pub(crate) fn position(&self) -> usize {
        self.pos
    }

    pub(crate) fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        let end = self.pos.checked_add(n).ok_or(GgufError::Overflow)?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or(GgufError::UnexpectedEof {
                offset: self.pos,
                needed: n,
                available: self.bytes.len().saturating_sub(self.pos),
            })?;
        self.pos = end;
        Ok(slice)
    }

    pub(crate) fn array<const N: usize>(&mut self) -> Result<[u8; N], GgufError> {
        let slice = self.read_bytes(N)?;
        let mut out = [0u8; N];
        out.copy_from_slice(slice);
        Ok(out)
    }

    pub(crate) fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.array::<1>()?[0])
    }
    pub(crate) fn i8(&mut self) -> Result<i8, GgufError> {
        Ok(i8::from_le_bytes(self.array()?))
    }
    pub(crate) fn u16(&mut self) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(self.array()?))
    }
    pub(crate) fn i16(&mut self) -> Result<i16, GgufError> {
        Ok(i16::from_le_bytes(self.array()?))
    }
    pub(crate) fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.array()?))
    }
    pub(crate) fn i32(&mut self) -> Result<i32, GgufError> {
        Ok(i32::from_le_bytes(self.array()?))
    }
    pub(crate) fn f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_le_bytes(self.array()?))
    }
    pub(crate) fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.array()?))
    }
    pub(crate) fn i64(&mut self) -> Result<i64, GgufError> {
        Ok(i64::from_le_bytes(self.array()?))
    }
    pub(crate) fn f64(&mut self) -> Result<f64, GgufError> {
        Ok(f64::from_le_bytes(self.array()?))
    }
    pub(crate) fn bool(&mut self) -> Result<bool, GgufError> {
        Ok(self.u8()? != 0)
    }

    /// String GGUF: `u64` de comprimento + bytes UTF-8.
    pub(crate) fn gguf_string(&mut self) -> Result<String, GgufError> {
        let len = self.u64()?;
        let len = usize::try_from(len).map_err(|_| GgufError::Overflow)?;
        let bytes = self.read_bytes(len)?;
        core::str::from_utf8(bytes)
            .map(|s| s.to_owned())
            .map_err(|_| GgufError::InvalidUtf8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_scalars_little_endian() {
        let bytes = [0x01, 0x02, 0x03, 0x04, b'h', b'i'];
        let mut r = Reader::new(&bytes);
        assert_eq!(r.u32().unwrap(), 0x0403_0201);
        assert_eq!(r.read_bytes(2).unwrap(), b"hi");
    }

    #[test]
    fn out_of_bounds_is_error_not_panic() {
        let bytes = [0x00, 0x01];
        let mut r = Reader::new(&bytes);
        assert!(r.u32().is_err());
    }

    #[test]
    fn gguf_string_roundtrip() {
        // u64 len = 3, "abc"
        let mut bytes = vec![3, 0, 0, 0, 0, 0, 0, 0];
        bytes.extend_from_slice(b"abc");
        let mut r = Reader::new(&bytes);
        assert_eq!(r.gguf_string().unwrap(), "abc");
    }

    #[test]
    fn string_length_overflow_is_error() {
        // len = u64::MAX → não pode ler, erro (sem alocar)
        let bytes = [0xFF; 8];
        let mut r = Reader::new(&bytes);
        assert!(r.gguf_string().is_err());
    }

    #[test]
    fn reads_all_scalar_widths_little_endian() {
        // i8, u16, i16, i32, bool em sequência.
        let bytes = [
            0xFF, // i8 = -1
            0x01, 0x02, // u16 = 0x0201
            0xFF, 0xFF, // i16 = -1
            0x04, 0x03, 0x02, 0x01, // i32 = 0x01020304
            0x00, // bool = false
            0x01, // u8 = 1
        ];
        let mut r = Reader::new(&bytes);
        assert_eq!(r.i8().unwrap(), -1);
        assert_eq!(r.u16().unwrap(), 0x0201);
        assert_eq!(r.i16().unwrap(), -1);
        assert_eq!(r.i32().unwrap(), 0x0102_0304);
        assert!(!r.bool().unwrap());
        assert_eq!(r.u8().unwrap(), 1);
        assert_eq!(r.position(), bytes.len());
    }

    #[test]
    fn reads_64bit_and_float_scalars() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1.5f32.to_le_bytes());
        bytes.extend_from_slice(&2.5f64.to_le_bytes());
        bytes.extend_from_slice(&0x0102_0304_0506_0708u64.to_le_bytes());
        bytes.extend_from_slice(&(-1i64).to_le_bytes());
        let mut r = Reader::new(&bytes);
        assert_eq!(r.f32().unwrap(), 1.5);
        assert_eq!(r.f64().unwrap(), 2.5);
        assert_eq!(r.u64().unwrap(), 0x0102_0304_0506_0708);
        assert_eq!(r.i64().unwrap(), -1);
    }
}
