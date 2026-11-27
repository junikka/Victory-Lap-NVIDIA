// PE import-table reader shared by the stand-in DLL tests.
// Included from each crate's integration test. Parses the file itself:
// no dumpbin, no network.

#[test]
fn import_parser_finds_ntdll_in_kernel32() {
    let imports = pe_imports(std::path::Path::new(r"C:\Windows\System32\kernel32.dll"));
    assert!(
        imports
            .iter()
            .any(|entry| entry.dll.eq_ignore_ascii_case("ntdll.dll")),
        "parser missed ntdll.dll\n{}",
        format_imports(&imports)
    );
}

struct PeImport {
    dll: String,
    symbols: Vec<String>,
}

fn assert_console_imports(dll: &std::path::Path) {
    let imports = pe_imports(dll);
    assert!(
        !imports.is_empty(),
        "{} has no imports; the parser did not read the import table",
        dll.display()
    );
    let mut rejected = Vec::new();
    for entry in &imports {
        let name = entry.dll.to_ascii_lowercase();
        let allowed =
            name == "kernel32.dll" || name == "ntdll.dll" || name.starts_with("api-ms-win-core-");
        if !allowed || name.contains("bcrypt") {
            rejected.push(entry.dll.clone());
        }
    }
    assert!(
        rejected.is_empty(),
        "{} imports a module other than KERNEL32.dll, ntdll.dll, or api-ms-win-core-*: {}\n{}",
        dll.display(),
        rejected.join(", "),
        format_imports(&imports)
    );
    let prng: Vec<&str> = imports
        .iter()
        .filter(|entry| {
            entry
                .symbols
                .iter()
                .any(|symbol| symbol.eq_ignore_ascii_case("ProcessPrng"))
        })
        .map(|entry| entry.dll.as_str())
        .collect();
    assert!(
        prng.is_empty(),
        "ProcessPrng is imported from {prng:?}\n{}",
        format_imports(&imports)
    );
}

fn format_imports(imports: &[PeImport]) -> String {
    let mut text = String::new();
    for entry in imports {
        text.push_str(&entry.dll);
        text.push('\n');
        for symbol in &entry.symbols {
            text.push_str("  ");
            text.push_str(symbol);
            text.push('\n');
        }
    }
    text
}

fn pe_imports(path: &std::path::Path) -> Vec<PeImport> {
    let bytes = std::fs::read(path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    parse_pe_imports(&bytes).unwrap_or_else(|err| panic!("parse {}: {err}", path.display()))
}

struct Section {
    virtual_address: u32,
    virtual_size: u32,
    raw_offset: u32,
    raw_size: u32,
}

fn parse_pe_imports(bytes: &[u8]) -> Result<Vec<PeImport>, String> {
    if bytes.get(0..2) != Some(b"MZ") {
        return Err("not a PE file".into());
    }
    let pe = u32_at(bytes, 0x3C).ok_or("missing e_lfanew")? as usize;
    if bytes.get(pe..pe + 4) != Some(b"PE\0\0") {
        return Err("missing PE signature".into());
    }
    let coff = pe + 4;
    let section_count = u16_at(bytes, coff + 2).ok_or("missing section count")? as usize;
    let optional_size = u16_at(bytes, coff + 16).ok_or("missing optional header size")? as usize;
    let optional = coff + 20;
    if u16_at(bytes, optional) != Some(0x20B) {
        return Err("expected a PE32+ image".into());
    }
    let directory_count =
        u32_at(bytes, optional + 108).ok_or("missing data directory count")? as usize;
    let sections = read_sections(bytes, optional + optional_size, section_count)?;
    let mut imports = read_import_table(bytes, &sections, optional, directory_count)?;
    imports.extend(read_delay_imports(
        bytes,
        &sections,
        optional,
        directory_count,
    )?);
    Ok(imports)
}

fn read_sections(bytes: &[u8], start: usize, count: usize) -> Result<Vec<Section>, String> {
    let mut sections = Vec::with_capacity(count);
    for index in 0..count {
        let offset = start + index * 40;
        sections.push(Section {
            virtual_size: u32_at(bytes, offset + 8).ok_or("section virtual size")?,
            virtual_address: u32_at(bytes, offset + 12).ok_or("section virtual address")?,
            raw_size: u32_at(bytes, offset + 16).ok_or("section raw size")?,
            raw_offset: u32_at(bytes, offset + 20).ok_or("section raw offset")?,
        });
    }
    Ok(sections)
}

fn data_directory(
    bytes: &[u8],
    optional: usize,
    directory_count: usize,
    index: usize,
) -> Result<Option<(u32, u32)>, String> {
    if index >= directory_count {
        return Ok(None);
    }
    let offset = optional + 112 + index * 8;
    let rva = u32_at(bytes, offset).ok_or("data directory RVA")?;
    let size = u32_at(bytes, offset + 4).ok_or("data directory size")?;
    if rva == 0 || size == 0 {
        Ok(None)
    } else {
        Ok(Some((rva, size)))
    }
}

fn read_import_table(
    bytes: &[u8],
    sections: &[Section],
    optional: usize,
    directory_count: usize,
) -> Result<Vec<PeImport>, String> {
    let Some((rva, size)) = data_directory(bytes, optional, directory_count, 1)? else {
        return Ok(Vec::new());
    };
    let count = (size as usize) / 20;
    let mut imports = Vec::new();
    for index in 0..count {
        let entry = rva
            .checked_add((index * 20) as u32)
            .ok_or("import descriptor RVA overflow")?;
        let offset =
            rva_to_offset(sections, entry).ok_or("import descriptor is not in a section")?;
        let name_rva = u32_at(bytes, offset + 12).ok_or("import name RVA")?;
        if name_rva == 0 {
            return Ok(imports);
        }
        let dll = dll_name(bytes, sections, name_rva)?;
        let original = u32_at(bytes, offset).ok_or("import lookup RVA")?;
        let first = u32_at(bytes, offset + 16).ok_or("import address RVA")?;
        let thunk = if original != 0 { original } else { first };
        let symbols = thunk_symbols(bytes, sections, thunk)?;
        imports.push(PeImport { dll, symbols });
    }
    Err("import table is not NUL-terminated".into())
}

fn read_delay_imports(
    bytes: &[u8],
    sections: &[Section],
    optional: usize,
    directory_count: usize,
) -> Result<Vec<PeImport>, String> {
    let Some((rva, size)) = data_directory(bytes, optional, directory_count, 13)? else {
        return Ok(Vec::new());
    };
    let count = (size as usize) / 32;
    let mut imports = Vec::new();
    for index in 0..count {
        let entry = rva
            .checked_add((index * 32) as u32)
            .ok_or("delay descriptor RVA overflow")?;
        let offset =
            rva_to_offset(sections, entry).ok_or("delay descriptor is not in a section")?;
        let attrs = u32_at(bytes, offset).ok_or("delay attributes")?;
        let name_rva = u32_at(bytes, offset + 4).ok_or("delay name RVA")?;
        if name_rva == 0 {
            return Ok(imports);
        }
        if attrs & 1 == 0 {
            return Err(format!(
                "delay import is not RVA-based (attributes {attrs:#x})"
            ));
        }
        let dll = dll_name(bytes, sections, name_rva)?;
        let int_rva = u32_at(bytes, offset + 16).ok_or("delay INT RVA")?;
        let symbols = thunk_symbols(bytes, sections, int_rva)?;
        imports.push(PeImport { dll, symbols });
    }
    Err("delay import table is not NUL-terminated".into())
}

fn dll_name(bytes: &[u8], sections: &[Section], rva: u32) -> Result<String, String> {
    let offset = rva_to_offset(sections, rva).ok_or_else(|| format!("DLL name RVA {rva:#x}"))?;
    cstr_at(bytes, offset)
}

fn thunk_symbols(
    bytes: &[u8],
    sections: &[Section],
    thunk_rva: u32,
) -> Result<Vec<String>, String> {
    if thunk_rva == 0 {
        return Ok(Vec::new());
    }
    let mut symbols = Vec::new();
    for index in 0..4096 {
        let rva = thunk_rva
            .checked_add(index * 8)
            .ok_or("import thunk RVA overflow")?;
        let offset = rva_to_offset(sections, rva).ok_or("import thunk is not in a section")?;
        let entry = u64_at(bytes, offset).ok_or("import thunk entry")?;
        if entry == 0 {
            return Ok(symbols);
        }
        symbols.push(import_symbol(bytes, sections, entry)?);
    }
    Err("import thunk has more than 4096 symbols".into())
}

fn import_symbol(bytes: &[u8], sections: &[Section], entry: u64) -> Result<String, String> {
    const ORDINAL: u64 = 1 << 63;
    if entry & ORDINAL != 0 {
        return Ok(format!("#{}", entry & 0xFFFF));
    }
    let rva = u32::try_from(entry).map_err(|_| format!("import entry {entry:#x} is not an RVA"))?;
    let offset = rva_to_offset(sections, rva)
        .ok_or_else(|| format!("import hint RVA {rva:#x} is not in a section"))?;
    let name = offset.checked_add(2).ok_or("import hint overflows")?;
    cstr_at(bytes, name)
}

fn cstr_at(bytes: &[u8], offset: usize) -> Result<String, String> {
    let rest = bytes
        .get(offset..)
        .ok_or("name runs off the end of the file")?;
    let end = rest
        .iter()
        .position(|&byte| byte == 0)
        .ok_or("name is not NUL-terminated")?;
    let name = &rest[..end];
    if name.is_empty() {
        return Err("name is empty".into());
    }
    if !name
        .iter()
        .all(|byte| byte.is_ascii() && !byte.is_ascii_control())
    {
        return Err("name is not printable ASCII".into());
    }
    Ok(String::from_utf8(name.to_vec()).expect("ASCII"))
}

fn rva_to_offset(sections: &[Section], rva: u32) -> Option<usize> {
    let section = sections.iter().find(|section| {
        let span = section.virtual_size.max(section.raw_size);
        rva >= section.virtual_address && rva - section.virtual_address < span
    })?;
    let delta = rva - section.virtual_address;
    if delta >= section.raw_size {
        return None;
    }
    Some((section.raw_offset + delta) as usize)
}

fn u16_at(bytes: &[u8], offset: usize) -> Option<u16> {
    let end = offset.checked_add(2)?;
    Some(u16::from_le_bytes(bytes.get(offset..end)?.try_into().ok()?))
}

fn u32_at(bytes: &[u8], offset: usize) -> Option<u32> {
    let end = offset.checked_add(4)?;
    Some(u32::from_le_bytes(bytes.get(offset..end)?.try_into().ok()?))
}

fn u64_at(bytes: &[u8], offset: usize) -> Option<u64> {
    let end = offset.checked_add(8)?;
    Some(u64::from_le_bytes(bytes.get(offset..end)?.try_into().ok()?))
}
