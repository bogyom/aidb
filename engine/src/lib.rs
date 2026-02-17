use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::sync::RwLock;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

mod catalog;
mod record;
mod slt;
mod sql;

pub use catalog::{Catalog, ColumnSchema, SqlType, TableSchema};
pub use record::{RowRecord, Value};
pub use slt::SltDatabase;

const HEADER_SIZE: usize = 32;
const MAGIC_BYTES: [u8; 8] = *b"AIDBDB\0\0";
const CURRENT_VERSION: u32 = 1;
const DEFAULT_PAGE_SIZE: u32 = 4096;
const NO_PAGE: u32 = u32::MAX;
const CATALOG_MAX_STRING: usize = u16::MAX as usize;
pub const PAGE_SIZE: usize = DEFAULT_PAGE_SIZE as usize;
pub const PAGE_HEADER_SIZE: usize = 16;
pub const PAGE_PAYLOAD_SIZE: usize = PAGE_SIZE - PAGE_HEADER_SIZE;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeapRowLocation {
    pub page_id: u32,
    pub offset: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnMeta {
    pub name: String,
    pub sql_type: SqlType,
}

impl ColumnMeta {
    pub fn new(name: impl Into<String>, sql_type: SqlType) -> Self {
        Self {
            name: name.into(),
            sql_type,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QueryResult {
    pub columns: Vec<ColumnMeta>,
    pub rows: Vec<Vec<Value>>,
}

impl QueryResult {
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        let columns = self
            .columns
            .iter()
            .map(|col| {
                serde_json::json!({
                    "name": col.name,
                    "type": format!("{:?}", col.sql_type),
                })
            })
            .collect::<Vec<_>>();
        let rows = self
            .rows
            .iter()
            .map(|row| row.iter().map(value_to_json).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        serde_json::json!({
            "columns": columns,
            "rows": rows,
        })
    }
}

fn value_to_json(value: &Value) -> serde_json::Value {
    match value {
        Value::Null => serde_json::Value::Null,
        Value::Integer(value) => serde_json::json!(value),
        Value::Real(value) => serde_json::json!(value),
        Value::Text(value) => serde_json::json!(value),
        Value::Boolean(value) => serde_json::json!(value),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    InvalidPath,
    InvalidSql,
    InvalidHeader,
    InvalidPageHeader,
    InvalidPageType(u8),
    PageOutOfBounds(u32),
    CatalogCorrupt,
    CatalogTooLarge,
    TableAlreadyExists,
    TableNotFound,
    RowTooLarge,
    RowCorrupt,
    UnsupportedVersion(u32),
    Closed,
    Io(std::io::ErrorKind),
}

pub type EngineResult<T> = Result<T, EngineError>;

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileHeader {
    version: u32,
    page_size: u32,
    free_list_head: u32,
    catalog_root: u32,
}

impl FileHeader {
    fn new(version: u32, page_size: u32, free_list_head: u32, catalog_root: u32) -> Self {
        Self {
            version,
            page_size,
            free_list_head,
            catalog_root,
        }
    }

    fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..8].copy_from_slice(&MAGIC_BYTES);
        bytes[8..12].copy_from_slice(&self.version.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.page_size.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.free_list_head.to_le_bytes());
        bytes[20..24].copy_from_slice(&self.catalog_root.to_le_bytes());
        bytes
    }

    fn from_bytes(bytes: [u8; HEADER_SIZE]) -> EngineResult<Self> {
        if bytes[0..8] != MAGIC_BYTES {
            return Err(EngineError::InvalidHeader);
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != CURRENT_VERSION {
            return Err(EngineError::UnsupportedVersion(version));
        }
        let page_size = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let free_list_head = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        Ok(Self {
            version,
            page_size,
            free_list_head,
            catalog_root: u32::from_le_bytes(bytes[20..24].try_into().unwrap()),
        })
    }
}

/// Page types used in the storage layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PageType {
    Free = 0,
    Catalog = 1,
    Heap = 2,
    Index = 3,
}

impl PageType {
    fn from_byte(value: u8) -> EngineResult<Self> {
        match value {
            0 => Ok(PageType::Free),
            1 => Ok(PageType::Catalog),
            2 => Ok(PageType::Heap),
            3 => Ok(PageType::Index),
            other => Err(EngineError::InvalidPageType(other)),
        }
    }
}

/// Page header layout as documented in `engine/README.md` under "Page Layout".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageHeader {
    pub page_type: PageType,
    pub flags: u8,
    pub page_id: u32,
    pub next_page: u32,
    pub payload_size: u32,
}

impl PageHeader {
    pub fn new(page_type: PageType, page_id: u32) -> Self {
        Self {
            page_type,
            flags: 0,
            page_id,
            next_page: 0,
            payload_size: 0,
        }
    }

    pub fn to_bytes(self) -> [u8; PAGE_HEADER_SIZE] {
        let mut bytes = [0u8; PAGE_HEADER_SIZE];
        bytes[0] = self.page_type as u8;
        bytes[1] = self.flags;
        bytes[2..4].copy_from_slice(&0u16.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.page_id.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.next_page.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.payload_size.to_le_bytes());
        bytes
    }

    pub fn from_bytes(bytes: [u8; PAGE_HEADER_SIZE]) -> EngineResult<Self> {
        let page_type = PageType::from_byte(bytes[0])?;
        let flags = bytes[1];
        let page_id = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let next_page = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let payload_size = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        if payload_size as usize > PAGE_PAYLOAD_SIZE {
            return Err(EngineError::InvalidPageHeader);
        }
        Ok(Self {
            page_type,
            flags,
            page_id,
            next_page,
            payload_size,
        })
    }
}

#[derive(Debug)]
pub struct Database {
    path: String,
    closed: bool,
    catalog: Catalog,
    table_pages: HashMap<String, Vec<u32>>,
    /// Concurrency policy: single writer, multiple readers. Writers block readers.
    /// Methods requiring `&mut self` are already exclusive via Rust borrowing, while
    /// shared (`&self`) operations take this lock to coordinate readers and writers.
    concurrency: RwLock<()>,
}

impl Database {
    pub fn open(path: impl Into<String>) -> EngineResult<Self> {
        let path = path.into();
        if path.trim().is_empty() {
            return Err(EngineError::InvalidPath);
        }
        let mut file = File::open(Path::new(&path))?;
        let header = read_file_header(&mut file)?;
        let catalog = if header.catalog_root == NO_PAGE {
            Catalog::new()
        } else {
            read_catalog_from_file(&mut file, header.catalog_root)?
        };
        let mut table_pages = HashMap::new();
        for table in catalog.tables() {
            table_pages.insert(table.name.clone(), Vec::new());
        }
        Ok(Self {
            path,
            closed: false,
            catalog,
            table_pages,
            concurrency: RwLock::new(()),
        })
    }

    pub fn create(path: impl Into<String>) -> EngineResult<Self> {
        let path = path.into();
        if path.trim().is_empty() {
            return Err(EngineError::InvalidPath);
        }
        write_header(&path)?;
        let mut db = Self {
            path,
            closed: false,
            catalog: Catalog::new(),
            table_pages: HashMap::new(),
            concurrency: RwLock::new(()),
        };
        db.initialize_catalog_unlocked()?;
        Ok(db)
    }

    fn read_guard(&self) -> std::sync::RwLockReadGuard<'_, ()> {
        self.concurrency.read().expect("concurrency lock")
    }

    fn write_guard(&self) -> std::sync::RwLockWriteGuard<'_, ()> {
        self.concurrency.write().expect("concurrency lock")
    }

    #[cfg(test)]
    pub(crate) fn write_guard_for_test(&self) -> std::sync::RwLockWriteGuard<'_, ()> {
        self.write_guard()
    }

    pub fn execute(&mut self, sql: &str) -> EngineResult<QueryResult> {
        self.execute_unlocked(sql)
    }

    fn execute_unlocked(&mut self, sql: &str) -> EngineResult<QueryResult> {
        if self.closed {
            return Err(EngineError::Closed);
        }
        if sql.trim().is_empty() {
            return Err(EngineError::InvalidSql);
        }
        let statements = crate::sql::parser::parse(sql).map_err(|_| EngineError::InvalidSql)?;
        let mut last_result = QueryResult::empty();
        for statement in &statements {
            crate::sql::validator::validate_statement(statement, &self.catalog)
                .map_err(map_validation_error)?;
            match statement {
                crate::sql::parser::Statement::CreateTable(create) => {
                    let table = table_from_ast(create)?;
                    self.create_table_unlocked(table)?;
                }
                crate::sql::parser::Statement::DropTable(drop) => {
                    self.drop_table_unlocked(&drop.name)?;
                }
                crate::sql::parser::Statement::Insert(insert) => {
                    let row = row_from_insert(insert, &self.catalog)?;
                    self.insert_table_row_unlocked(&insert.table, &row)?;
                }
                crate::sql::parser::Statement::Select(_) => {
                    let select = match statement {
                        crate::sql::parser::Statement::Select(select) => select,
                        _ => unreachable!("select statement"),
                    };
                    last_result = crate::sql::interpreter::execute_select(select, self)?;
                }
            }
        }
        Ok(last_result)
    }

    pub fn close(mut self) -> EngineResult<()> {
        self.close_unlocked()
    }

    fn close_unlocked(&mut self) -> EngineResult<()> {
        if self.closed {
            return Err(EngineError::Closed);
        }
        self.closed = true;
        Ok(())
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    fn table_pages_unlocked(&self, table: &str) -> Option<&[u32]> {
        self.table_pages.get(table).map(|pages| pages.as_slice())
    }

    #[allow(dead_code)]
    pub(crate) fn scan_table_rows(&self, table: &str) -> EngineResult<Vec<RowRecord>> {
        let _guard = self.read_guard();
        self.scan_table_rows_unlocked(table)
    }

    pub(crate) fn scan_table_rows_unlocked(&self, table: &str) -> EngineResult<Vec<RowRecord>> {
        let pages = match self.table_pages_unlocked(table) {
            Some(pages) => pages,
            None => return Ok(Vec::new()),
        };
        let mut rows = Vec::new();
        for page_id in pages {
            let page_rows = self.scan_heap_page_unlocked(*page_id)?;
            rows.extend(page_rows);
        }
        Ok(rows)
    }

    pub fn create_table(&mut self, table: TableSchema) -> EngineResult<()> {
        self.create_table_unlocked(table)
    }

    fn create_table_unlocked(&mut self, table: TableSchema) -> EngineResult<()> {
        if self.closed {
            return Err(EngineError::Closed);
        }
        if self.catalog.table(&table.name).is_some() {
            return Err(EngineError::TableAlreadyExists);
        }
        let table_name = table.name.clone();
        self.catalog.add_table(table);
        self.persist_catalog_unlocked()?;
        self.table_pages.entry(table_name).or_default();
        Ok(())
    }

    pub fn drop_table(&mut self, name: &str) -> EngineResult<()> {
        self.drop_table_unlocked(name)
    }

    fn drop_table_unlocked(&mut self, name: &str) -> EngineResult<()> {
        if self.closed {
            return Err(EngineError::Closed);
        }
        if self.catalog.remove_table(name).is_none() {
            return Err(EngineError::TableNotFound);
        }
        self.persist_catalog_unlocked()?;
        self.table_pages.remove(name);
        Ok(())
    }

    pub fn allocate_page(&self, page_type: PageType) -> EngineResult<u32> {
        let _guard = self.write_guard();
        self.allocate_page_unlocked(page_type)
    }

    fn allocate_page_unlocked(&self, page_type: PageType) -> EngineResult<u32> {
        if self.closed {
            return Err(EngineError::Closed);
        }
        let mut file = open_rw(&self.path)?;
        let mut header = read_file_header(&mut file)?;

        if header.free_list_head != NO_PAGE {
            let page_id = header.free_list_head;
            let free_header = read_page_header(&mut file, page_id)?;
            header.free_list_head = free_header.next_page;
            write_file_header(&mut file, header)?;
            let new_header = PageHeader::new(page_type, page_id);
            write_page_header(&mut file, page_id, new_header)?;
            return Ok(page_id);
        }

        let page_id = page_count(&file)?;
        let page_header = PageHeader::new(page_type, page_id);
        write_page_header(&mut file, page_id, page_header)?;
        write_page_payload_zeroed(&mut file)?;
        Ok(page_id)
    }

    pub fn free_page(&self, page_id: u32) -> EngineResult<()> {
        let _guard = self.write_guard();
        self.free_page_unlocked(page_id)
    }

    fn free_page_unlocked(&self, page_id: u32) -> EngineResult<()> {
        if self.closed {
            return Err(EngineError::Closed);
        }
        let mut file = open_rw(&self.path)?;
        let page_count = page_count(&file)?;
        if page_id >= page_count {
            return Err(EngineError::PageOutOfBounds(page_id));
        }
        let mut file_header = read_file_header(&mut file)?;
        let mut header = read_page_header(&mut file, page_id)?;
        if header.page_type == PageType::Free {
            return Ok(());
        }
        header.page_type = PageType::Free;
        header.flags = 0;
        header.next_page = file_header.free_list_head;
        header.payload_size = 0;
        write_page_header(&mut file, page_id, header)?;
        file_header.free_list_head = page_id;
        write_file_header(&mut file, file_header)?;
        Ok(())
    }

    pub fn read_page_header(&self, page_id: u32) -> EngineResult<PageHeader> {
        let _guard = self.read_guard();
        self.read_page_header_unlocked(page_id)
    }

    fn read_page_header_unlocked(&self, page_id: u32) -> EngineResult<PageHeader> {
        if self.closed {
            return Err(EngineError::Closed);
        }
        let mut file = open_rw(&self.path)?;
        let page_count = page_count(&file)?;
        if page_id >= page_count {
            return Err(EngineError::PageOutOfBounds(page_id));
        }
        read_page_header(&mut file, page_id)
    }

    pub fn insert_heap_row(&self, row: &RowRecord) -> EngineResult<HeapRowLocation> {
        let _guard = self.write_guard();
        self.insert_heap_row_unlocked(row)
    }

    fn insert_heap_row_unlocked(&self, row: &RowRecord) -> EngineResult<HeapRowLocation> {
        if self.closed {
            return Err(EngineError::Closed);
        }
        let row_bytes = row.encode()?;
        if row_bytes.len() > PAGE_PAYLOAD_SIZE {
            return Err(EngineError::RowTooLarge);
        }

        let mut file = open_rw(&self.path)?;
        let page_count = page_count(&file)?;
        for page_id in 0..page_count {
            let header = read_page_header(&mut file, page_id)?;
            if header.page_type != PageType::Heap {
                continue;
            }
            let used = header.payload_size as usize;
            if used > PAGE_PAYLOAD_SIZE {
                return Err(EngineError::InvalidPageHeader);
            }
            let free = PAGE_PAYLOAD_SIZE - used;
            if row_bytes.len() <= free {
                let offset = used;
                write_page_payload_at(&mut file, page_id, offset, &row_bytes)?;
                let mut updated = header;
                updated.payload_size = (used + row_bytes.len()) as u32;
                write_page_header(&mut file, page_id, updated)?;
                return Ok(HeapRowLocation {
                    page_id,
                    offset: offset as u32,
                });
            }
        }
        drop(file);

        let page_id = self.allocate_page_unlocked(PageType::Heap)?;
        let mut file = open_rw(&self.path)?;
        write_page_payload_at(&mut file, page_id, 0, &row_bytes)?;
        let mut header = read_page_header(&mut file, page_id)?;
        header.payload_size = row_bytes.len() as u32;
        write_page_header(&mut file, page_id, header)?;
        Ok(HeapRowLocation { page_id, offset: 0 })
    }

    pub fn insert_table_row(&mut self, table: &str, row: &RowRecord) -> EngineResult<()> {
        self.insert_table_row_unlocked(table, row)
    }

    fn insert_table_row_unlocked(&mut self, table: &str, row: &RowRecord) -> EngineResult<()> {
        if self.closed {
            return Err(EngineError::Closed);
        }
        if self.catalog.table(table).is_none() {
            return Err(EngineError::TableNotFound);
        }
        let row_bytes = row.encode()?;
        if row_bytes.len() > PAGE_PAYLOAD_SIZE {
            return Err(EngineError::RowTooLarge);
        }

        {
            let pages = self.table_pages.entry(table.to_string()).or_default();
            let mut file = open_rw(&self.path)?;
            for page_id in pages.iter() {
                let header = read_page_header(&mut file, *page_id)?;
                if header.page_type != PageType::Heap {
                    return Err(EngineError::InvalidPageHeader);
                }
                let used = header.payload_size as usize;
                if used > PAGE_PAYLOAD_SIZE {
                    return Err(EngineError::InvalidPageHeader);
                }
                let free = PAGE_PAYLOAD_SIZE - used;
                if row_bytes.len() <= free {
                    let offset = used;
                    write_page_payload_at(&mut file, *page_id, offset, &row_bytes)?;
                    let mut updated = header;
                    updated.payload_size = (used + row_bytes.len()) as u32;
                    write_page_header(&mut file, *page_id, updated)?;
                    return Ok(());
                }
            }
        }

        let page_id = self.allocate_page_unlocked(PageType::Heap)?;
        let mut file = open_rw(&self.path)?;
        write_page_payload_at(&mut file, page_id, 0, &row_bytes)?;
        let mut header = read_page_header(&mut file, page_id)?;
        header.payload_size = row_bytes.len() as u32;
        write_page_header(&mut file, page_id, header)?;
        let pages = self.table_pages.entry(table.to_string()).or_default();
        pages.push(page_id);
        Ok(())
    }

    pub fn read_heap_row(&self, location: HeapRowLocation) -> EngineResult<RowRecord> {
        let _guard = self.read_guard();
        self.read_heap_row_unlocked(location)
    }

    fn read_heap_row_unlocked(&self, location: HeapRowLocation) -> EngineResult<RowRecord> {
        if self.closed {
            return Err(EngineError::Closed);
        }
        let mut file = open_rw(&self.path)?;
        let page_count = page_count(&file)?;
        if location.page_id >= page_count {
            return Err(EngineError::PageOutOfBounds(location.page_id));
        }
        let header = read_page_header(&mut file, location.page_id)?;
        if header.page_type != PageType::Heap {
            return Err(EngineError::InvalidPageHeader);
        }
        let used = header.payload_size as usize;
        let offset = location.offset as usize;
        if offset + 4 > used {
            return Err(EngineError::RowCorrupt);
        }
        let len_bytes = read_page_payload_at(&mut file, location.page_id, offset, 4)?;
        let row_len = u32::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
        if offset + row_len > used {
            return Err(EngineError::RowCorrupt);
        }
        let row_bytes = read_page_payload_at(&mut file, location.page_id, offset, row_len)?;
        RowRecord::decode(&row_bytes).map_err(|_| EngineError::RowCorrupt)
    }

    pub fn scan_heap_page(&self, page_id: u32) -> EngineResult<Vec<RowRecord>> {
        let _guard = self.read_guard();
        self.scan_heap_page_unlocked(page_id)
    }

    fn scan_heap_page_unlocked(&self, page_id: u32) -> EngineResult<Vec<RowRecord>> {
        if self.closed {
            return Err(EngineError::Closed);
        }
        let mut file = open_rw(&self.path)?;
        let page_count = page_count(&file)?;
        if page_id >= page_count {
            return Err(EngineError::PageOutOfBounds(page_id));
        }
        let header = read_page_header(&mut file, page_id)?;
        if header.page_type != PageType::Heap {
            return Err(EngineError::InvalidPageHeader);
        }
        let used = header.payload_size as usize;
        if used > PAGE_PAYLOAD_SIZE {
            return Err(EngineError::InvalidPageHeader);
        }
        if used == 0 {
            return Ok(Vec::new());
        }

        let payload = read_page_payload(&mut file, page_id, used)?;
        let mut rows = Vec::new();
        let mut cursor = 0usize;
        while cursor < used {
            if cursor + 4 > used {
                return Err(EngineError::RowCorrupt);
            }
            let row_len =
                u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap()) as usize;
            if row_len == 0 || cursor + row_len > used {
                return Err(EngineError::RowCorrupt);
            }
            let row_bytes = &payload[cursor..cursor + row_len];
            let row = RowRecord::decode(row_bytes).map_err(|_| EngineError::RowCorrupt)?;
            rows.push(row);
            cursor += row_len;
        }
        Ok(rows)
    }

    fn initialize_catalog_unlocked(&mut self) -> EngineResult<()> {
        let page_id = self.allocate_page_unlocked(PageType::Catalog)?;
        let mut file = open_rw(&self.path)?;
        let mut header = read_file_header(&mut file)?;
        header.catalog_root = page_id;
        write_file_header(&mut file, header)?;
        write_catalog_to_file(&mut file, page_id, &self.catalog)?;
        Ok(())
    }

    fn persist_catalog_unlocked(&self) -> EngineResult<()> {
        let mut file = open_rw(&self.path)?;
        let header = read_file_header(&mut file)?;
        let page_id = if header.catalog_root == NO_PAGE {
            drop(file);
            let page_id = self.allocate_page_unlocked(PageType::Catalog)?;
            let mut file = open_rw(&self.path)?;
            let mut header = read_file_header(&mut file)?;
            header.catalog_root = page_id;
            write_file_header(&mut file, header)?;
            write_catalog_to_file(&mut file, page_id, &self.catalog)?;
            return Ok(());
        } else {
            header.catalog_root
        };
        write_catalog_to_file(&mut file, page_id, &self.catalog)?;
        Ok(())
    }
}

fn table_from_ast(create: &crate::sql::parser::CreateTable) -> EngineResult<TableSchema> {
    let mut columns = Vec::with_capacity(create.columns.len());
    let mut primary_key = None;

    for column in &create.columns {
        let sql_type = match column.data_type {
            crate::sql::parser::TypeName::Integer => SqlType::Integer,
            crate::sql::parser::TypeName::Real => SqlType::Real,
            crate::sql::parser::TypeName::Text => SqlType::Text,
            crate::sql::parser::TypeName::Boolean => SqlType::Boolean,
        };
        let nullable = !column.primary_key;
        columns.push(ColumnSchema::new(&column.name, sql_type, nullable));
        if column.primary_key {
            if primary_key.replace(column.name.clone()).is_some() {
                return Err(EngineError::InvalidSql);
            }
        }
    }

    let mut table = TableSchema::new(&create.name, columns);
    if let Some(primary_key) = primary_key {
        table = table.with_primary_key(primary_key);
    }
    Ok(table)
}

fn row_from_insert(
    insert: &crate::sql::parser::Insert,
    catalog: &Catalog,
) -> EngineResult<RowRecord> {
    let table = catalog
        .table(&insert.table)
        .ok_or(EngineError::InvalidSql)?;
    let mut values = vec![Value::Null; table.columns.len()];

    let column_names: Vec<String> = if let Some(columns) = &insert.columns {
        columns.clone()
    } else {
        table.columns.iter().map(|col| col.name.clone()).collect()
    };

    if column_names.len() != insert.values.len() {
        return Err(EngineError::InvalidSql);
    }

    for (name, expr) in column_names.iter().zip(insert.values.iter()) {
        let index = table
            .columns
            .iter()
            .position(|column| column.name == *name)
            .ok_or(EngineError::InvalidSql)?;
        values[index] = expr_to_value(expr)?;
    }

    Ok(RowRecord::new(values))
}

fn expr_to_value(expr: &crate::sql::parser::Expr) -> EngineResult<Value> {
    match expr {
        crate::sql::parser::Expr::Literal(literal) => literal_to_value(literal),
        _ => Err(EngineError::InvalidSql),
    }
}

fn literal_to_value(literal: &crate::sql::lexer::Literal) -> EngineResult<Value> {
    match literal {
        crate::sql::lexer::Literal::Number(raw) => {
            if raw.contains('.') {
                let value = raw.parse::<f64>().map_err(|_| EngineError::InvalidSql)?;
                Ok(Value::Real(value))
            } else {
                let value = raw.parse::<i64>().map_err(|_| EngineError::InvalidSql)?;
                Ok(Value::Integer(value))
            }
        }
        crate::sql::lexer::Literal::String(value) => Ok(Value::Text(value.clone())),
        crate::sql::lexer::Literal::Boolean(value) => Ok(Value::Boolean(*value)),
        crate::sql::lexer::Literal::Null => Ok(Value::Null),
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        self.closed = true;
    }
}

fn map_validation_error(error: crate::sql::validator::ValidationError) -> EngineError {
    match error {
        crate::sql::validator::ValidationError::TableNotFound(_) => EngineError::TableNotFound,
        crate::sql::validator::ValidationError::TableAlreadyExists(_) => {
            EngineError::TableAlreadyExists
        }
        _ => EngineError::InvalidSql,
    }
}

fn write_header(path: &str) -> EngineResult<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .read(true)
        .open(Path::new(path))?;
    let header = FileHeader::new(CURRENT_VERSION, DEFAULT_PAGE_SIZE, NO_PAGE, NO_PAGE);
    file.write_all(&header.to_bytes())?;
    file.flush()?;
    Ok(())
}

fn read_file_header(file: &mut File) -> EngineResult<FileHeader> {
    file.seek(SeekFrom::Start(0))?;
    let mut buffer = [0u8; HEADER_SIZE];
    file.read_exact(&mut buffer)?;
    FileHeader::from_bytes(buffer)
}

fn write_file_header(file: &mut File, header: FileHeader) -> EngineResult<()> {
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&header.to_bytes())?;
    Ok(())
}

fn open_rw(path: &str) -> EngineResult<File> {
    Ok(OpenOptions::new().read(true).write(true).open(Path::new(path))?)
}

fn page_offset(page_id: u32) -> u64 {
    HEADER_SIZE as u64 + (page_id as u64) * (PAGE_SIZE as u64)
}

fn page_count(file: &File) -> EngineResult<u32> {
    let len = file.metadata()?.len();
    if len < HEADER_SIZE as u64 {
        return Err(EngineError::InvalidHeader);
    }
    let remaining = len - HEADER_SIZE as u64;
    if remaining % PAGE_SIZE as u64 != 0 {
        return Err(EngineError::InvalidHeader);
    }
    Ok((remaining / PAGE_SIZE as u64) as u32)
}

fn read_page_header(file: &mut File, page_id: u32) -> EngineResult<PageHeader> {
    let offset = page_offset(page_id);
    file.seek(SeekFrom::Start(offset))?;
    let mut buffer = [0u8; PAGE_HEADER_SIZE];
    file.read_exact(&mut buffer)?;
    PageHeader::from_bytes(buffer)
}

fn write_page_header(file: &mut File, page_id: u32, header: PageHeader) -> EngineResult<()> {
    let offset = page_offset(page_id);
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(&header.to_bytes())?;
    Ok(())
}

fn write_page_payload_zeroed(file: &mut File) -> EngineResult<()> {
    let zeros = vec![0u8; PAGE_PAYLOAD_SIZE];
    file.write_all(&zeros)?;
    Ok(())
}

fn write_page_payload(file: &mut File, page_id: u32, payload: &[u8]) -> EngineResult<()> {
    if payload.len() > PAGE_PAYLOAD_SIZE {
        return Err(EngineError::CatalogTooLarge);
    }
    let offset = page_offset(page_id) + PAGE_HEADER_SIZE as u64;
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(payload)?;
    let remaining = PAGE_PAYLOAD_SIZE - payload.len();
    if remaining > 0 {
        let zeros = vec![0u8; remaining];
        file.write_all(&zeros)?;
    }
    Ok(())
}

fn read_page_payload(file: &mut File, page_id: u32, size: usize) -> EngineResult<Vec<u8>> {
    if size > PAGE_PAYLOAD_SIZE {
        return Err(EngineError::InvalidPageHeader);
    }
    let offset = page_offset(page_id) + PAGE_HEADER_SIZE as u64;
    file.seek(SeekFrom::Start(offset))?;
    let mut buffer = vec![0u8; size];
    file.read_exact(&mut buffer)?;
    Ok(buffer)
}

fn write_page_payload_at(
    file: &mut File,
    page_id: u32,
    offset: usize,
    payload: &[u8],
) -> EngineResult<()> {
    if offset + payload.len() > PAGE_PAYLOAD_SIZE {
        return Err(EngineError::RowTooLarge);
    }
    let base = page_offset(page_id) + PAGE_HEADER_SIZE as u64 + offset as u64;
    file.seek(SeekFrom::Start(base))?;
    file.write_all(payload)?;
    Ok(())
}

fn read_page_payload_at(
    file: &mut File,
    page_id: u32,
    offset: usize,
    size: usize,
) -> EngineResult<Vec<u8>> {
    if offset + size > PAGE_PAYLOAD_SIZE {
        return Err(EngineError::InvalidPageHeader);
    }
    let base = page_offset(page_id) + PAGE_HEADER_SIZE as u64 + offset as u64;
    file.seek(SeekFrom::Start(base))?;
    let mut buffer = vec![0u8; size];
    file.read_exact(&mut buffer)?;
    Ok(buffer)
}

fn write_catalog_to_file(
    file: &mut File,
    page_id: u32,
    catalog: &Catalog,
) -> EngineResult<()> {
    let payload = encode_catalog(catalog)?;
    if payload.len() > PAGE_PAYLOAD_SIZE {
        return Err(EngineError::CatalogTooLarge);
    }
    let mut header = PageHeader::new(PageType::Catalog, page_id);
    header.payload_size = payload.len() as u32;
    write_page_header(file, page_id, header)?;
    write_page_payload(file, page_id, &payload)?;
    Ok(())
}

fn read_catalog_from_file(file: &mut File, page_id: u32) -> EngineResult<Catalog> {
    let header = read_page_header(file, page_id)?;
    if header.page_type != PageType::Catalog {
        return Err(EngineError::InvalidPageHeader);
    }
    let payload = read_page_payload(file, page_id, header.payload_size as usize)?;
    decode_catalog(&payload)
}

fn encode_catalog(catalog: &Catalog) -> EngineResult<Vec<u8>> {
    let mut bytes = Vec::new();
    let table_count = catalog.tables().len();
    bytes.extend_from_slice(&(table_count as u32).to_le_bytes());

    for table in catalog.tables() {
        encode_string(&table.name, &mut bytes)?;
        let column_count = table.columns.len();
        if column_count > u16::MAX as usize {
            return Err(EngineError::CatalogTooLarge);
        }
        bytes.extend_from_slice(&(column_count as u16).to_le_bytes());
        match &table.primary_key {
            Some(pk) => {
                let len = pk.as_bytes().len();
                if len >= u16::MAX as usize {
                    return Err(EngineError::CatalogTooLarge);
                }
                bytes.extend_from_slice(&(len as u16).to_le_bytes());
                bytes.extend_from_slice(pk.as_bytes());
            }
            None => {
                bytes.extend_from_slice(&u16::MAX.to_le_bytes());
            }
        }
        for column in &table.columns {
            encode_string(&column.name, &mut bytes)?;
            bytes.push(sql_type_to_byte(column.sql_type));
            bytes.push(if column.nullable { 1 } else { 0 });
        }
    }

    Ok(bytes)
}

fn decode_catalog(bytes: &[u8]) -> EngineResult<Catalog> {
    let mut cursor = 0usize;
    let table_count = read_u32(bytes, &mut cursor)? as usize;
    let mut catalog = Catalog::new();
    for _ in 0..table_count {
        let name = read_string(bytes, &mut cursor)?;
        let column_count = read_u16(bytes, &mut cursor)? as usize;
        let pk_len = read_u16(bytes, &mut cursor)?;
        let primary_key = if pk_len == u16::MAX {
            None
        } else {
            let pk = read_string_with_len(bytes, &mut cursor, pk_len as usize)?;
            Some(pk)
        };

        let mut columns = Vec::with_capacity(column_count);
        for _ in 0..column_count {
            let column_name = read_string(bytes, &mut cursor)?;
            let sql_type = sql_type_from_byte(read_u8(bytes, &mut cursor)?)?;
            let nullable = read_u8(bytes, &mut cursor)? != 0;
            columns.push(ColumnSchema::new(column_name, sql_type, nullable));
        }
        let mut table = TableSchema::new(name, columns);
        if let Some(pk) = primary_key {
            table = table.with_primary_key(pk);
        }
        catalog.add_table(table);
    }
    Ok(catalog)
}

fn encode_string(value: &str, bytes: &mut Vec<u8>) -> EngineResult<()> {
    let len = value.as_bytes().len();
    if len > CATALOG_MAX_STRING {
        return Err(EngineError::CatalogTooLarge);
    }
    bytes.extend_from_slice(&(len as u16).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
}

fn read_string(bytes: &[u8], cursor: &mut usize) -> EngineResult<String> {
    let len = read_u16(bytes, cursor)? as usize;
    read_string_with_len(bytes, cursor, len)
}

fn read_string_with_len(bytes: &[u8], cursor: &mut usize, len: usize) -> EngineResult<String> {
    let slice = take(bytes, cursor, len)?;
    std::str::from_utf8(slice)
        .map(|s| s.to_string())
        .map_err(|_| EngineError::CatalogCorrupt)
}

fn read_u8(bytes: &[u8], cursor: &mut usize) -> EngineResult<u8> {
    let slice = take(bytes, cursor, 1)?;
    Ok(slice[0])
}

fn read_u16(bytes: &[u8], cursor: &mut usize) -> EngineResult<u16> {
    let slice = take(bytes, cursor, 2)?;
    Ok(u16::from_le_bytes(slice.try_into().unwrap()))
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> EngineResult<u32> {
    let slice = take(bytes, cursor, 4)?;
    Ok(u32::from_le_bytes(slice.try_into().unwrap()))
}

fn take<'a>(bytes: &'a [u8], cursor: &mut usize, len: usize) -> EngineResult<&'a [u8]> {
    if *cursor + len > bytes.len() {
        return Err(EngineError::CatalogCorrupt);
    }
    let start = *cursor;
    let end = start + len;
    *cursor = end;
    Ok(&bytes[start..end])
}

fn sql_type_to_byte(sql_type: SqlType) -> u8 {
    match sql_type {
        SqlType::Integer => 0,
        SqlType::Real => 1,
        SqlType::Text => 2,
        SqlType::Boolean => 3,
    }
}

fn sql_type_from_byte(value: u8) -> EngineResult<SqlType> {
    match value {
        0 => Ok(SqlType::Integer),
        1 => Ok(SqlType::Real),
        2 => Ok(SqlType::Text),
        3 => Ok(SqlType::Boolean),
        _ => Err(EngineError::CatalogCorrupt),
    }
}

impl From<std::io::Error> for EngineError {
    fn from(error: std::io::Error) -> Self {
        EngineError::Io(error.kind())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_db_path(label: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        path.push(format!("aidb_{label}_{nanos}.db"));
        path
    }

    #[test]
    fn open_requires_non_empty_path() {
        let err = Database::open("   ").unwrap_err();
        assert_eq!(err, EngineError::InvalidPath);
    }

    #[test]
    fn create_writes_header() {
        let path = temp_db_path("create_writes_header");
        let db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        assert_eq!(db.path(), path.to_string_lossy().as_ref());

        let bytes = fs::read(&path).expect("read header");
        assert!(bytes.len() >= HEADER_SIZE);
        assert_eq!(&bytes[0..8], &MAGIC_BYTES);
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        assert_eq!(version, CURRENT_VERSION);
    }

    #[test]
    fn open_validates_magic_bytes() {
        let path = temp_db_path("open_invalid_magic");
        let mut file = File::create(&path).expect("create file");
        file.write_all(b"NOTAIDB").expect("write magic");
        file.write_all(&[0u8; HEADER_SIZE]).expect("pad");
        drop(file);

        let err = Database::open(path.to_string_lossy().as_ref()).unwrap_err();
        assert_eq!(err, EngineError::InvalidHeader);
    }

    #[test]
    fn open_validates_version() {
        let path = temp_db_path("open_invalid_version");
        let mut file = File::create(&path).expect("create file");
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..8].copy_from_slice(&MAGIC_BYTES);
        bytes[8..12].copy_from_slice(&99u32.to_le_bytes());
        file.write_all(&bytes).expect("write header");
        drop(file);

        let err = Database::open(path.to_string_lossy().as_ref()).unwrap_err();
        assert_eq!(err, EngineError::UnsupportedVersion(99));
    }

    #[test]
    fn execute_rejects_empty_sql() {
        let path = temp_db_path("execute_empty_sql");
        let mut db =
            Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        let err = db.execute("   ").unwrap_err();
        assert_eq!(err, EngineError::InvalidSql);
    }

    #[test]
    fn execute_select_with_filter_order_limit() {
        let path = temp_db_path("execute_select_filter_order_limit");
        let mut db =
            Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        db.execute("CREATE TABLE users (id INTEGER, name TEXT)")
            .expect("create table");
        db.execute("INSERT INTO users (id, name) VALUES (2, 'Bob')")
            .expect("insert row");
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Ada')")
            .expect("insert row");
        db.execute("INSERT INTO users (id, name) VALUES (3, 'Cal')")
            .expect("insert row");

        let result = db
            .execute(
                "SELECT name FROM users WHERE id = 1 OR id = 3 ORDER BY id DESC LIMIT 1",
            )
            .expect("select");

        assert_eq!(
            result.columns,
            vec![ColumnMeta::new("name", SqlType::Text)]
        );
        assert_eq!(result.rows, vec![vec![Value::Text("Cal".to_string())]]);
    }

    #[test]
    fn execute_count_star_returns_total_rows() {
        let path = temp_db_path("execute_count_star");
        let mut db =
            Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        db.execute("CREATE TABLE users (id INTEGER, name TEXT)")
            .expect("create table");
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Ada')")
            .expect("insert row");
        db.execute("INSERT INTO users (id, name) VALUES (2, 'Bob')")
            .expect("insert row");
        db.execute("INSERT INTO users (id, name) VALUES (3, 'Cal')")
            .expect("insert row");

        let result = db.execute("SELECT COUNT(*) FROM users").expect("select");
        assert_eq!(
            result.columns,
            vec![ColumnMeta::new("expr", SqlType::Integer)]
        );
        assert_eq!(result.rows, vec![vec![Value::Integer(3)]]);
    }

    #[test]
    fn execute_count_in_scalar_subquery() {
        let path = temp_db_path("execute_count_subquery");
        let mut db =
            Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        db.execute("CREATE TABLE users (id INTEGER)")
            .expect("create table");
        db.execute("INSERT INTO users (id) VALUES (1)")
            .expect("insert row");
        db.execute("INSERT INTO users (id) VALUES (2)")
            .expect("insert row");

        let result = db
            .execute("SELECT (SELECT COUNT(*) FROM users)")
            .expect("select");
        assert_eq!(result.rows, vec![vec![Value::Integer(2)]]);
    }

    #[test]
    fn execute_correlated_scalar_subquery_per_outer_row() {
        let path = temp_db_path("execute_correlated_scalar_subquery");
        let mut db =
            Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        db.execute("CREATE TABLE users (id INTEGER)")
            .expect("create table");
        db.execute("INSERT INTO users (id) VALUES (1)")
            .expect("insert row");
        db.execute("INSERT INTO users (id) VALUES (2)")
            .expect("insert row");
        db.execute("INSERT INTO users (id) VALUES (3)")
            .expect("insert row");

        let result = db
            .execute(
                "SELECT id, (SELECT COUNT(*) FROM users u2 WHERE u2.id <= users.id) \
                 FROM users ORDER BY id",
            )
            .expect("select");

        assert_eq!(
            result.rows,
            vec![
                vec![Value::Integer(1), Value::Integer(1)],
                vec![Value::Integer(2), Value::Integer(2)],
                vec![Value::Integer(3), Value::Integer(3)],
            ]
        );
    }

    #[test]
    fn execute_exists_returns_true_when_rows_present() {
        let path = temp_db_path("execute_exists_rows_present");
        let mut db =
            Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        db.execute("CREATE TABLE users (id INTEGER)")
            .expect("create table");
        db.execute("INSERT INTO users (id) VALUES (1)")
            .expect("insert row");

        let result = db
            .execute("SELECT EXISTS (SELECT id FROM users)")
            .expect("select");
        assert_eq!(result.rows, vec![vec![Value::Boolean(true)]]);
    }

    #[test]
    fn execute_exists_with_correlation_filters_rows() {
        let path = temp_db_path("execute_exists_correlated");
        let mut db =
            Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        db.execute("CREATE TABLE users (id INTEGER)")
            .expect("create table");
        db.execute("INSERT INTO users (id) VALUES (1)")
            .expect("insert row");
        db.execute("INSERT INTO users (id) VALUES (2)")
            .expect("insert row");
        db.execute("INSERT INTO users (id) VALUES (3)")
            .expect("insert row");

        let result = db
            .execute(
                "SELECT u1.id, \
                 EXISTS (SELECT 1 FROM users u2 WHERE u2.id = u1.id AND u2.id > 1) \
                 FROM users u1 ORDER BY u1.id",
            )
            .expect("select");

        assert_eq!(
            result.rows,
            vec![
                vec![Value::Integer(1), Value::Boolean(false)],
                vec![Value::Integer(2), Value::Boolean(true)],
                vec![Value::Integer(3), Value::Boolean(true)],
            ]
        );
    }

    #[test]
    fn execute_select_scans_multiple_heap_pages() {
        let path = temp_db_path("execute_select_scan_multiple_pages");
        let mut db =
            Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        db.execute("CREATE TABLE t (id INTEGER, name TEXT)")
            .expect("create table");

        let large_len = PAGE_PAYLOAD_SIZE - 40;
        let large_text = "x".repeat(large_len);
        let sql = format!("INSERT INTO t (id, name) VALUES (1, '{}')", large_text);
        db.execute(&sql).expect("insert large row");
        db.execute("INSERT INTO t (id, name) VALUES (2, 'small')")
            .expect("insert second row");

        let result = db
            .execute("SELECT id FROM t WHERE id = 2 ORDER BY id ASC")
            .expect("select");

        assert_eq!(result.rows, vec![vec![Value::Integer(2)]]);
    }

    #[test]
    fn execute_insert_rejects_unknown_table() {
        let path = temp_db_path("execute_insert_unknown_table");
        let mut db =
            Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        let err = db
            .execute("INSERT INTO missing (id) VALUES (1)")
            .unwrap_err();
        assert_eq!(err, EngineError::TableNotFound);
    }

    #[test]
    fn execute_insert_rejects_unknown_column() {
        let path = temp_db_path("execute_insert_unknown_column");
        let mut db =
            Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        db.execute("CREATE TABLE users (id INTEGER)")
            .expect("create table");
        let err = db
            .execute("INSERT INTO users (missing) VALUES (1)")
            .unwrap_err();
        assert_eq!(err, EngineError::InvalidSql);
    }

    #[test]
    fn execute_insert_defaults_null_for_missing_columns() {
        let path = temp_db_path("execute_insert_defaults_null");
        let mut db =
            Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        db.execute("CREATE TABLE users (id INTEGER, name TEXT)")
            .expect("create table");
        db.execute("INSERT INTO users (id) VALUES (1)")
            .expect("insert row");
        let result = db
            .execute("SELECT name FROM users WHERE id = 1")
            .expect("select");
        assert_eq!(result.rows, vec![vec![Value::Null]]);
    }

    #[test]
    fn query_result_to_json_includes_types() {
        let result = QueryResult {
            columns: vec![ColumnMeta::new("id", SqlType::Integer)],
            rows: vec![vec![Value::Integer(7)]],
        };
        let json = result.to_json();
        assert_eq!(
            json,
            serde_json::json!({
                "columns": [{ "name": "id", "type": "Integer" }],
                "rows": [[7]],
            })
        );
    }

    #[test]
    fn concurrency_blocks_reads_during_writes() {
        use std::sync::{mpsc, Arc};
        use std::thread;
        use std::time::Duration;

        let path = temp_db_path("concurrency_blocks_reads");
        let db = Arc::new(Database::create(path.to_string_lossy().as_ref()).expect("create"));
        let page_id = db
            .allocate_page(PageType::Heap)
            .expect("allocate page");

        let (locked_tx, locked_rx) = mpsc::channel();
        let (read_done_tx, read_done_rx) = mpsc::channel();

        let db_writer = Arc::clone(&db);
        let writer = thread::spawn(move || {
            let _guard = db_writer.write_guard_for_test();
            locked_tx.send(()).expect("signal locked");
            thread::sleep(Duration::from_millis(200));
        });

        let db_reader = Arc::clone(&db);
        let reader = thread::spawn(move || {
            locked_rx.recv().expect("wait for lock");
            db_reader
                .read_page_header(page_id)
                .expect("read header");
            read_done_tx.send(()).expect("signal read done");
        });

        assert!(read_done_rx.recv_timeout(Duration::from_millis(50)).is_err());
        read_done_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("read completes after write");

        writer.join().expect("writer join");
        reader.join().expect("reader join");
    }

    #[test]
    fn close_consumes_handle() {
        let path = temp_db_path("close_consumes_handle");
        let db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        assert!(db.close().is_ok());
    }

    #[test]
    fn page_header_round_trip() {
        let header = PageHeader {
            page_type: PageType::Heap,
            flags: 1,
            page_id: 42,
            next_page: 7,
            payload_size: 128,
        };
        let bytes = header.to_bytes();
        let decoded = PageHeader::from_bytes(bytes).expect("decode page header");
        assert_eq!(decoded, header);
    }

    #[test]
    fn page_header_rejects_unknown_type() {
        let mut bytes = [0u8; PAGE_HEADER_SIZE];
        bytes[0] = 9;
        let err = PageHeader::from_bytes(bytes).unwrap_err();
        assert_eq!(err, EngineError::InvalidPageType(9));
    }

    #[test]
    fn page_header_rejects_payload_overflow() {
        let mut bytes = [0u8; PAGE_HEADER_SIZE];
        bytes[0] = PageType::Heap as u8;
        bytes[12..16].copy_from_slice(&((PAGE_PAYLOAD_SIZE + 1) as u32).to_le_bytes());
        let err = PageHeader::from_bytes(bytes).unwrap_err();
        assert_eq!(err, EngineError::InvalidPageHeader);
    }

    #[test]
    fn allocate_page_persists_header() {
        let path = temp_db_path("allocate_page_persists_header");
        let db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        let page_id = db.allocate_page(PageType::Heap).expect("allocate page");
        assert_eq!(page_id, 1);

        let header = db.read_page_header(page_id).expect("read header");
        assert_eq!(header.page_type, PageType::Heap);
        assert_eq!(header.page_id, page_id);

        let len = fs::metadata(&path).expect("metadata").len();
        assert_eq!(len, (HEADER_SIZE + (2 * PAGE_SIZE)) as u64);
    }

    #[test]
    fn free_page_marks_as_free() {
        let path = temp_db_path("free_page_marks_as_free");
        let db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        let page_id = db.allocate_page(PageType::Catalog).expect("allocate page");
        db.free_page(page_id).expect("free page");
        let header = db.read_page_header(page_id).expect("read header");
        assert_eq!(header.page_type, PageType::Free);
    }

    #[test]
    fn allocator_reuses_freed_pages() {
        let path = temp_db_path("allocator_reuses_freed_pages");
        let db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        let page_id = db.allocate_page(PageType::Heap).expect("allocate page");
        db.free_page(page_id).expect("free page");
        let reused_id = db.allocate_page(PageType::Catalog).expect("allocate page");
        assert_eq!(reused_id, page_id);
        let header = db.read_page_header(reused_id).expect("read header");
        assert_eq!(header.page_type, PageType::Catalog);
    }

    #[test]
    fn free_list_persists_across_restart() {
        let path = temp_db_path("free_list_persists");
        let db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        let page_id = db.allocate_page(PageType::Heap).expect("allocate page");
        db.free_page(page_id).expect("free page");
        drop(db);

        let reopened = Database::open(path.to_string_lossy().as_ref()).expect("reopen");
        let reused_id = reopened
            .allocate_page(PageType::Catalog)
            .expect("allocate page");
        assert_eq!(reused_id, page_id);
    }

    #[test]
    fn free_list_head_updates_on_free() {
        let path = temp_db_path("free_list_head_updates");
        let db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        let first = db.allocate_page(PageType::Heap).expect("allocate page");
        let second = db.allocate_page(PageType::Heap).expect("allocate page");
        db.free_page(first).expect("free page");
        db.free_page(second).expect("free page");

        let mut file = open_rw(path.to_string_lossy().as_ref()).expect("open file");
        let header = read_file_header(&mut file).expect("read header");
        assert_eq!(header.free_list_head, second);
    }

    #[test]
    fn catalog_persists_across_restart() {
        let path = temp_db_path("catalog_persists");
        let mut db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        db.create_table(
            TableSchema::new(
                "people",
                vec![
                    ColumnSchema::new("id", SqlType::Integer, false),
                    ColumnSchema::new("name", SqlType::Text, false),
                ],
            )
            .with_primary_key("id"),
        )
        .expect("create table");
        drop(db);

        let reopened = Database::open(path.to_string_lossy().as_ref()).expect("reopen");
        let table = reopened.catalog().table("people").expect("table exists");
        assert_eq!(table.columns.len(), 2);
        assert_eq!(table.primary_key.as_deref(), Some("id"));
    }

    #[test]
    fn catalog_drop_persists() {
        let path = temp_db_path("catalog_drop_persists");
        let mut db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        db.create_table(TableSchema::new(
            "items",
            vec![ColumnSchema::new("sku", SqlType::Text, false)],
        ))
        .expect("create table");
        db.drop_table("items").expect("drop table");
        drop(db);

        let reopened = Database::open(path.to_string_lossy().as_ref()).expect("reopen");
        assert!(reopened.catalog().table("items").is_none());
    }

    #[test]
    fn heap_row_written_and_persisted() {
        let path = temp_db_path("heap_row_written");
        let db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        let row = RowRecord::new(vec![
            Value::Integer(7),
            Value::Text("alpha".to_string()),
            Value::Boolean(true),
        ]);
        let location = db.insert_heap_row(&row).expect("insert row");
        assert_eq!(location.page_id, 1);
        drop(db);

        let reopened = Database::open(path.to_string_lossy().as_ref()).expect("reopen");
        let loaded = reopened
            .read_heap_row(location)
            .expect("read heap row");
        assert_eq!(loaded, row);
    }

    #[test]
    fn heap_page_tracks_free_space() {
        let path = temp_db_path("heap_page_free_space");
        let db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        let row1 = RowRecord::new(vec![Value::Text("one".to_string())]);
        let row2 = RowRecord::new(vec![Value::Text("two".to_string())]);
        let loc1 = db.insert_heap_row(&row1).expect("insert row1");
        let loc2 = db.insert_heap_row(&row2).expect("insert row2");
        assert_eq!(loc1.page_id, loc2.page_id);

        let header = db
            .read_page_header(loc1.page_id)
            .expect("read header");
        let expected = row1.encoded_size().unwrap() + row2.encoded_size().unwrap();
        assert_eq!(header.payload_size as usize, expected);
    }

    #[test]
    fn heap_page_overflow_allocates_new_page() {
        let path = temp_db_path("heap_page_overflow");
        let db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        let large_text = "a".repeat(PAGE_PAYLOAD_SIZE - 16);
        let row1 = RowRecord::new(vec![Value::Text(large_text)]);
        let row2 = RowRecord::new(vec![Value::Text("small".to_string())]);
        let loc1 = db.insert_heap_row(&row1).expect("insert row1");
        let loc2 = db.insert_heap_row(&row2).expect("insert row2");
        assert_ne!(loc1.page_id, loc2.page_id);
        assert_eq!(loc2.page_id, loc1.page_id + 1);
    }

    #[test]
    fn heap_page_scan_reads_rows() {
        let path = temp_db_path("heap_page_scan_reads_rows");
        let db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        let row1 = RowRecord::new(vec![Value::Integer(1)]);
        let row2 = RowRecord::new(vec![Value::Integer(2)]);
        let loc1 = db.insert_heap_row(&row1).expect("insert row1");
        let loc2 = db.insert_heap_row(&row2).expect("insert row2");
        assert_eq!(loc1.page_id, loc2.page_id);

        let rows = db.scan_heap_page(loc1.page_id).expect("scan");
        assert_eq!(rows, vec![row1, row2]);
    }

    #[test]
    fn heap_page_scan_handles_empty_page() {
        let path = temp_db_path("heap_page_scan_handles_empty");
        let db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        let page_id = db.allocate_page(PageType::Heap).expect("allocate page");
        let rows = db.scan_heap_page(page_id).expect("scan");
        assert!(rows.is_empty());
    }

    #[test]
    fn heap_page_scan_respects_partial_fill() {
        let path = temp_db_path("heap_page_scan_partial");
        let db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        let row = RowRecord::new(vec![Value::Text("partial".to_string())]);
        let loc = db.insert_heap_row(&row).expect("insert row");
        let rows = db.scan_heap_page(loc.page_id).expect("scan");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0], row);
    }
}
