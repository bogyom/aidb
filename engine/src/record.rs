use crate::EngineError;

const TYPE_INTEGER: u8 = 0;
const TYPE_REAL: u8 = 1;
const TYPE_TEXT: u8 = 2;
const TYPE_BOOLEAN: u8 = 3;
const TYPE_NULL: u8 = 255;

const RECORD_HEADER_SIZE: usize = 8;
const COLUMN_META_SIZE: usize = 8;

/// Row record layout (documented in `engine/README.md`).
#[derive(Debug, Clone, PartialEq)]
pub struct RowRecord {
    pub values: Vec<Value>,
}

impl RowRecord {
    pub fn new(values: Vec<Value>) -> Self {
        Self { values }
    }

    pub fn encoded_size(&self) -> Result<usize, EngineError> {
        let column_count = self.values.len();
        if column_count > u16::MAX as usize {
            return Err(EngineError::RowTooLarge);
        }
        let mut size = RECORD_HEADER_SIZE + column_count * COLUMN_META_SIZE;
        for value in &self.values {
            size = size
                .checked_add(value.payload_len()?)
                .ok_or(EngineError::RowTooLarge)?;
        }
        if size > u32::MAX as usize {
            return Err(EngineError::RowTooLarge);
        }
        Ok(size)
    }

    pub fn encode(&self) -> Result<Vec<u8>, EngineError> {
        let total_len = self.encoded_size()?;
        let column_count = self.values.len() as u16;
        let mut bytes = Vec::with_capacity(total_len);
        bytes.extend_from_slice(&(total_len as u32).to_le_bytes());
        bytes.extend_from_slice(&column_count.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());

        for value in &self.values {
            let (type_tag, flags, length) = value.meta()?;
            bytes.push(type_tag);
            bytes.push(flags);
            bytes.extend_from_slice(&0u16.to_le_bytes());
            bytes.extend_from_slice(&(length as u32).to_le_bytes());
        }

        for value in &self.values {
            value.write_payload(&mut bytes)?;
        }

        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, EngineError> {
        if bytes.len() < RECORD_HEADER_SIZE {
            return Err(EngineError::RowCorrupt);
        }
        let mut cursor = 0usize;
        let total_len = read_u32(bytes, &mut cursor)? as usize;
        if total_len != bytes.len() {
            return Err(EngineError::RowCorrupt);
        }
        let column_count = read_u16(bytes, &mut cursor)? as usize;
        cursor += 2; // reserved
        let meta_size = RECORD_HEADER_SIZE + column_count * COLUMN_META_SIZE;
        if bytes.len() < meta_size {
            return Err(EngineError::RowCorrupt);
        }

        let mut metas = Vec::with_capacity(column_count);
        for _ in 0..column_count {
            let type_tag = read_u8(bytes, &mut cursor)?;
            let flags = read_u8(bytes, &mut cursor)?;
            cursor += 2;
            let length = read_u32(bytes, &mut cursor)? as usize;
            metas.push((type_tag, flags, length));
        }

        let mut values = Vec::with_capacity(column_count);
        for (type_tag, flags, length) in metas {
            let payload = take(bytes, &mut cursor, length)?;
            let value = Value::from_parts(type_tag, flags, payload)?;
            values.push(value);
        }

        if cursor != bytes.len() {
            return Err(EngineError::RowCorrupt);
        }

        Ok(Self { values })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
    Boolean(bool),
}

impl Value {
    fn payload_len(&self) -> Result<usize, EngineError> {
        Ok(match self {
            Value::Null => 0,
            Value::Integer(_) => 8,
            Value::Real(_) => 8,
            Value::Boolean(_) => 1,
            Value::Text(value) => value.as_bytes().len(),
        })
    }

    fn meta(&self) -> Result<(u8, u8, usize), EngineError> {
        let (type_tag, flags) = match self {
            Value::Null => (TYPE_NULL, 1),
            Value::Integer(_) => (TYPE_INTEGER, 0),
            Value::Real(_) => (TYPE_REAL, 0),
            Value::Text(_) => (TYPE_TEXT, 0),
            Value::Boolean(_) => (TYPE_BOOLEAN, 0),
        };
        let length = self.payload_len()?;
        Ok((type_tag, flags, length))
    }

    fn write_payload(&self, out: &mut Vec<u8>) -> Result<(), EngineError> {
        match self {
            Value::Null => Ok(()),
            Value::Integer(value) => {
                out.extend_from_slice(&value.to_le_bytes());
                Ok(())
            }
            Value::Real(value) => {
                out.extend_from_slice(&value.to_le_bytes());
                Ok(())
            }
            Value::Text(value) => {
                out.extend_from_slice(value.as_bytes());
                Ok(())
            }
            Value::Boolean(value) => {
                out.push(if *value { 1 } else { 0 });
                Ok(())
            }
        }
    }

    fn from_parts(type_tag: u8, flags: u8, payload: &[u8]) -> Result<Self, EngineError> {
        match type_tag {
            TYPE_NULL => {
                if payload.is_empty() && (flags & 1) == 1 {
                    Ok(Value::Null)
                } else {
                    Err(EngineError::RowCorrupt)
                }
            }
            TYPE_INTEGER => {
                if payload.len() != 8 {
                    return Err(EngineError::RowCorrupt);
                }
                let value = i64::from_le_bytes(payload.try_into().unwrap());
                Ok(Value::Integer(value))
            }
            TYPE_REAL => {
                if payload.len() != 8 {
                    return Err(EngineError::RowCorrupt);
                }
                let value = f64::from_le_bytes(payload.try_into().unwrap());
                Ok(Value::Real(value))
            }
            TYPE_TEXT => {
                let value = std::str::from_utf8(payload)
                    .map_err(|_| EngineError::RowCorrupt)?
                    .to_string();
                Ok(Value::Text(value))
            }
            TYPE_BOOLEAN => {
                if payload.len() != 1 {
                    return Err(EngineError::RowCorrupt);
                }
                Ok(Value::Boolean(payload[0] != 0))
            }
            _ => Err(EngineError::RowCorrupt),
        }
    }
}

fn read_u8(bytes: &[u8], cursor: &mut usize) -> Result<u8, EngineError> {
    let slice = take(bytes, cursor, 1)?;
    Ok(slice[0])
}

fn read_u16(bytes: &[u8], cursor: &mut usize) -> Result<u16, EngineError> {
    let slice = take(bytes, cursor, 2)?;
    Ok(u16::from_le_bytes(slice.try_into().unwrap()))
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, EngineError> {
    let slice = take(bytes, cursor, 4)?;
    Ok(u32::from_le_bytes(slice.try_into().unwrap()))
}

fn take<'a>(bytes: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8], EngineError> {
    if *cursor + len > bytes.len() {
        return Err(EngineError::RowCorrupt);
    }
    let start = *cursor;
    let end = start + len;
    *cursor = end;
    Ok(&bytes[start..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_record_round_trip() {
        let record = RowRecord::new(vec![
            Value::Integer(42),
            Value::Real(3.5),
            Value::Text("hello".to_string()),
            Value::Boolean(true),
            Value::Null,
        ]);
        let bytes = record.encode().expect("encode");
        let decoded = RowRecord::decode(&bytes).expect("decode");
        assert_eq!(decoded, record);
        assert_eq!(bytes.len(), record.encoded_size().unwrap());
    }

    #[test]
    fn row_record_rejects_corrupt_length() {
        let record = RowRecord::new(vec![Value::Boolean(true)]);
        let mut bytes = record.encode().expect("encode");
        let offset = RECORD_HEADER_SIZE;
        bytes[offset + 4..offset + 8].copy_from_slice(&2u32.to_le_bytes());
        let err = RowRecord::decode(&bytes).unwrap_err();
        assert_eq!(err, EngineError::RowCorrupt);
    }
}
