# Engine Storage Header

The database file begins with a fixed-size header used to identify the file
format, check compatibility, and capture basic layout metadata.

## Header layout (32 bytes)

- Bytes 0-7: magic bytes `AIDBDB\0\0`
- Bytes 8-11: format version (little-endian `u32`)
- Bytes 12-15: default page size in bytes (little-endian `u32`)
- Bytes 16-19: free list head page id (little-endian `u32`, `0xFFFF_FFFF` if none)
- Bytes 20-23: catalog root page id (little-endian `u32`, `0xFFFF_FFFF` if none)
- Bytes 24-31: reserved (zero-filled)

The engine writes this header on database creation and validates the magic
bytes and version on open.

## Page Layout

Each page is `4096` bytes with a fixed 16-byte header followed by payload.

### Page header layout (16 bytes)

- Byte 0: page type (see enum in `engine/src/lib.rs`)
- Byte 1: flags
- Bytes 2-3: reserved (zero-filled)
- Bytes 4-7: page id (little-endian `u32`)
- Bytes 8-11: next page id (little-endian `u32`, 0 if none)
- Bytes 12-15: payload size in bytes (little-endian `u32`)

Payload begins at byte 16 and spans the remainder of the page.

### Page types

- `Free`: available for reuse
- `Catalog`: system catalog pages
- `Heap`: table row storage
- `Index`: index storage

## Catalog Schema Model

The in-memory catalog uses `TableSchema` and `ColumnSchema` definitions
in `engine/src/catalog.rs`. The MVP SQL subset recognizes these types:

- `Integer` (signed 64-bit)
- `Real` (64-bit float)
- `Text` (UTF-8 string)
- `Boolean` (true/false)

## Row Record Layout

Each heap row is encoded as a record with a fixed header, per-column metadata,
and a payload section.

### Record header (8 bytes)

- Bytes 0-3: total record length in bytes (little-endian `u32`)
- Bytes 4-5: column count (little-endian `u16`)
- Bytes 6-7: reserved (zero-filled)

### Column metadata (8 bytes each)

- Byte 0: type tag (`0`=Integer, `1`=Real, `2`=Text, `3`=Boolean, `255`=Null)
- Byte 1: flags (bit 0 set if NULL)
- Bytes 2-3: reserved (zero-filled)
- Bytes 4-7: payload length in bytes (little-endian `u32`)

### Payload

Payload bytes for each column are stored sequentially in column order. Fixed
types use their native sizes (8 bytes for Integer/Real, 1 byte for Boolean).
Text uses its UTF-8 byte length; Null values have length 0.

## Heap Page Writing

Heap pages store row records sequentially in the payload region. The page
header `payload_size` tracks the used bytes so free space is
`PAGE_PAYLOAD_SIZE - payload_size`. Inserts append to the first heap page with
enough space; if none fit, a new heap page is allocated and the row is written
at offset 0.

Heap page scans read the payload from offset 0 up to `payload_size`, decoding
row records in sequence. Empty heap pages return no rows; partially filled
pages stop at the tracked payload boundary.
