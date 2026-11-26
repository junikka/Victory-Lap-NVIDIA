//! The DXBC container: a header, then a table of FourCC-tagged parts.
//!
//! ```text
//! 0x00  "DXBC"
//! 0x04  digest[16]
//! 0x14  version major u16, minor u16
//! 0x18  total size u32
//! 0x1C  part count u32
//! 0x20  part offsets u32[count]
//! part: fourcc[4], size u32, data[size]
//! ```

use crate::ShaderError;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FourCC(pub [u8; 4]);

impl FourCC {
    pub const DXIL: FourCC = FourCC(*b"DXIL");
    /// Native code another compiler made, in place of `DXIL` (VIctory upstream 0b35b7f).
    pub const ACCC: FourCC = FourCC(*b"ACCC");
    pub const STAT: FourCC = FourCC(*b"STAT");
    pub const RTS0: FourCC = FourCC(*b"RTS0");
    pub const PSV0: FourCC = FourCC(*b"PSV0");
    pub const ISG1: FourCC = FourCC(*b"ISG1");
    pub const OSG1: FourCC = FourCC(*b"OSG1");
    /// Per-primitive outputs of a mesh shader.
    pub const PSG1: FourCC = FourCC(*b"PSG1");
    pub const SFI0: FourCC = FourCC(*b"SFI0");
    pub const HASH: FourCC = FourCC(*b"HASH");
    pub const ILDB: FourCC = FourCC(*b"ILDB");
    pub const ILDN: FourCC = FourCC(*b"ILDN");
}

impl std::fmt::Debug for FourCC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl std::fmt::Display for FourCC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for &b in &self.0 {
            let c = if b.is_ascii_graphic() { b as char } else { '.' };
            write!(f, "{c}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Part<'a> {
    pub fourcc: FourCC,
    /// Offset of the part header within the container.
    pub offset: usize,
    pub data: &'a [u8],
}

#[derive(Debug, Clone)]
pub struct Container<'a> {
    pub digest: [u8; 16],
    pub version: (u16, u16),
    pub parts: Vec<Part<'a>>,
}

const HEADER_SIZE: usize = 0x20;
const MAX_PARTS: usize = 256;

impl<'a> Container<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self, ShaderError> {
        if data.len() < HEADER_SIZE || &data[0..4] != b"DXBC" {
            return Err(ShaderError::NotDxbc);
        }
        let digest = data[4..20].try_into().expect("16 bytes");
        let version = (u16_at(data, 0x14)?, u16_at(data, 0x16)?);
        let total = u32_at(data, 0x18)? as usize;
        if total > data.len() || total < HEADER_SIZE {
            return Err(ShaderError::Malformed("container size disagrees with data"));
        }
        let data = &data[..total];
        let count = u32_at(data, 0x1C)? as usize;
        if count > MAX_PARTS {
            return Err(ShaderError::Malformed("implausible part count"));
        }
        let mut parts = Vec::with_capacity(count);
        for i in 0..count {
            let offset = u32_at(data, HEADER_SIZE + 4 * i)? as usize;
            let fourcc = FourCC(
                data.get(offset..offset + 4)
                    .ok_or(ShaderError::Malformed("part header out of bounds"))?
                    .try_into()
                    .expect("4 bytes"),
            );
            let size = u32_at(data, offset + 4)? as usize;
            let body = data
                .get(offset + 8..offset + 8 + size)
                .ok_or(ShaderError::Malformed("part data out of bounds"))?;
            parts.push(Part {
                fourcc,
                offset,
                data: body,
            });
        }
        Ok(Container {
            digest,
            version,
            parts,
        })
    }

    pub fn part(&self, fourcc: FourCC) -> Option<&Part<'a>> {
        self.parts.iter().find(|p| p.fourcc == fourcc)
    }

    /// The FSPX compute module in this container's `ACCC` part, if it has one.
    pub fn fspx(&self) -> Option<Fspx<'a>> {
        Fspx::parse(self.part(FourCC::ACCC)?.data).ok()
    }
}

/// A precompiled compute module (`FSPX`): kept whole in a DXBC `ACCC` part, or
/// handed to the device as separate kinds. The translator's strict read of a
/// valid module is [`crate::native::fspx_program`]; this is the directory of
/// the same bytes, including a module that would not translate.
///
/// ```text
/// 0x00  "FSPX"
/// 0x04  version u32
/// 0x08  entry count u32
/// 0x0C  total size u32
/// 0x10  entries[count]: kind u32, offset u32, size u32, reserved u32
/// ```
///
/// Kinds kept: 0 (dispatch state), 1 (pipeline metadata), 6 (the GFX10
/// program) and 7 (resource and hash records). Others are skipped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fspx<'a> {
    pub version: u32,
    pub count: u32,
    pub total_size: u32,
    pub dispatch_state: Option<&'a [u8]>,
    pub kind1: Option<&'a [u8]>,
    pub program: Option<&'a [u8]>,
    pub records: Option<&'a [u8]>,
}

/// What an [`Fspx`]'s dispatch state (kind 0) says about how its program runs.
/// The bytes match [`crate::native::fspx_program`]: thread-group size at
/// `+0x0c` as three `u16`s, wave32 in bit 4 of the `u16` at `+0x12`, then
/// `COMPUTE_PGM_RSRC1` and `COMPUTE_PGM_RSRC2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchInfo {
    pub threads: [u32; 3],
    pub wave32: bool,
    pub rsrc1: u32,
    pub rsrc2: u32,
}

impl<'a> Fspx<'a> {
    pub const MAGIC: &'static [u8; 4] = b"FSPX";
    const HEADER_SIZE: usize = 0x10;
    const ENTRY_SIZE: usize = 16;
    /// Kind 0 is at least this long: it is read up to its `rsrc2` word.
    pub const DISPATCH_STATE_SIZE: usize = 28;
    const MAX_ENTRIES: usize = 64;

    pub fn parse(data: &'a [u8]) -> Result<Self, ShaderError> {
        if !data.starts_with(Self::MAGIC) {
            return Err(ShaderError::Malformed("not an FSPX module"));
        }
        let version = u32_at(data, 4)?;
        let count = u32_at(data, 8)?;
        let total_size = u32_at(data, 12)?;
        let total = total_size as usize;
        if total > data.len() || total < Self::HEADER_SIZE {
            return Err(ShaderError::Malformed("FSPX size disagrees with data"));
        }
        let data = &data[..total];
        if count as usize > Self::MAX_ENTRIES
            || Self::HEADER_SIZE + count as usize * Self::ENTRY_SIZE > total
        {
            return Err(ShaderError::Malformed("FSPX entry table out of bounds"));
        }
        let mut fspx = Fspx {
            version,
            count,
            total_size,
            dispatch_state: None,
            kind1: None,
            program: None,
            records: None,
        };
        for i in 0..count as usize {
            let at = Self::HEADER_SIZE + i * Self::ENTRY_SIZE;
            let kind = u32_at(data, at)?;
            let offset = u32_at(data, at + 4)? as usize;
            let size = u32_at(data, at + 8)? as usize;
            let body = offset
                .checked_add(size)
                .and_then(|end| data.get(offset..end))
                .ok_or(ShaderError::Malformed("FSPX entry out of bounds"))?;
            match kind {
                0 => fspx.dispatch_state = Some(body),
                1 => fspx.kind1 = Some(body),
                6 => fspx.program = Some(body),
                7 => fspx.records = Some(body),
                _ => {}
            }
        }
        Ok(fspx)
    }

    /// The module's thread group, wave size and resource words, from its
    /// dispatch state. `None` when that piece is missing or short.
    pub fn dispatch_info(&self) -> Option<DispatchInfo> {
        dispatch_info(self.dispatch_state?)
    }

    /// A module made of the four kinds, in that order after the entry table
    /// (version 1), as the device hands them over separately. `None` when the
    /// pieces do not fit in the `u32` size the header stores.
    pub fn assemble(kind0: &[u8], kind1: &[u8], kind6: &[u8], kind7: &[u8]) -> Option<Vec<u8>> {
        const KINDS: [u32; 4] = [0, 1, 6, 7];
        let bodies = [kind0, kind1, kind6, kind7];
        let table = Self::HEADER_SIZE + KINDS.len() * Self::ENTRY_SIZE;
        let body_len = bodies.iter().try_fold(0usize, |n, b| n.checked_add(b.len()))?;
        let total = table.checked_add(body_len)?;
        let total_u32 = u32::try_from(total).ok()?;
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(Self::MAGIC);
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(KINDS.len() as u32).to_le_bytes());
        out.extend_from_slice(&total_u32.to_le_bytes());
        let mut offset = table;
        for (kind, body) in KINDS.iter().zip(bodies) {
            let at = u32::try_from(offset).ok()?;
            let len = u32::try_from(body.len()).ok()?;
            out.extend_from_slice(&kind.to_le_bytes());
            out.extend_from_slice(&at.to_le_bytes());
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            offset += body.len();
        }
        for body in bodies {
            out.extend_from_slice(body);
        }
        Some(out)
    }
}

/// The dispatch state (kind 0) of an FSPX module.
pub fn dispatch_info(state: &[u8]) -> Option<DispatchInfo> {
    if state.len() < Fspx::DISPATCH_STATE_SIZE {
        return None;
    }
    let word = |at| u32_at(state, at).ok();
    let (w3, w4) = (word(0x0c)?, word(0x10)?);
    Some(DispatchInfo {
        threads: [w3 & 0xffff, w3 >> 16, w4 & 0xffff],
        wave32: w4 & (1 << 20) != 0,
        rsrc1: word(0x14)?,
        rsrc2: word(0x18)?,
    })
}

/// The FSPX module in `bytes`, if it is one bare or in the `ACCC` part of a
/// DXBC container.
pub fn inspect_fspx(bytes: &[u8]) -> Option<Fspx<'_>> {
    if bytes.starts_with(Fspx::MAGIC) {
        return Fspx::parse(bytes).ok();
    }
    Container::parse(bytes).ok()?.fspx()
}

pub(crate) fn u16_at(data: &[u8], at: usize) -> Result<u16, ShaderError> {
    data.get(at..at + 2)
        .map(|b| u16::from_le_bytes(b.try_into().expect("2 bytes")))
        .ok_or(ShaderError::Malformed("truncated"))
}

pub(crate) fn u32_at(data: &[u8], at: usize) -> Result<u32, ShaderError> {
    data.get(at..at + 4)
        .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
        .ok_or(ShaderError::Malformed("truncated"))
}

pub(crate) fn u64_at(data: &[u8], at: usize) -> Result<u64, ShaderError> {
    data.get(at..at + 8)
        .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
        .ok_or(ShaderError::Malformed("truncated"))
}

/// Builds a container from parts (test helper).
#[cfg(test)]
pub fn build(parts: &[(FourCC, &[u8])]) -> Vec<u8> {
    let table = HEADER_SIZE + 4 * parts.len();
    let total = table + parts.iter().map(|(_, d)| 8 + d.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"DXBC");
    out.extend_from_slice(&[0; 16]);
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(parts.len() as u32).to_le_bytes());
    let mut offset = table;
    for (_, d) in parts {
        out.extend_from_slice(&(offset as u32).to_le_bytes());
        offset += 8 + d.len();
    }
    for (fourcc, d) in parts {
        out.extend_from_slice(&fourcc.0);
        out.extend_from_slice(&(d.len() as u32).to_le_bytes());
        out.extend_from_slice(d);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_parts() {
        let bytes = build(&[(FourCC::SFI0, &[1, 2, 3, 4]), (FourCC::HASH, &[9; 20])]);
        let c = Container::parse(&bytes).unwrap();
        assert_eq!(c.parts.len(), 2);
        assert_eq!(c.part(FourCC::SFI0).unwrap().data, &[1, 2, 3, 4]);
        assert_eq!(c.part(FourCC::HASH).unwrap().data.len(), 20);
        assert!(c.part(FourCC::DXIL).is_none());
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(Container::parse(b"nope").unwrap_err(), ShaderError::NotDxbc);
        let mut bytes = build(&[(FourCC::SFI0, &[0; 8])]);
        let last = bytes.len() - 1;
        bytes[0x20] = 0xFF; // corrupt the first part offset
        assert!(Container::parse(&bytes).is_err());
        bytes.truncate(last);
        assert!(Container::parse(&bytes).is_err());
    }

    /// Kind 0, as [`crate::native::fspx_program`] reads it.
    fn dispatch_state(threads: [u16; 3], wave32: bool, rsrc1: u32, rsrc2: u32) -> [u8; 28] {
        let mut state = [0u8; 28];
        state[0x0c..0x0e].copy_from_slice(&threads[0].to_le_bytes());
        state[0x0e..0x10].copy_from_slice(&threads[1].to_le_bytes());
        state[0x10..0x12].copy_from_slice(&threads[2].to_le_bytes());
        let flags: u16 = if wave32 { 1 << 4 } else { 0 };
        state[0x12..0x14].copy_from_slice(&flags.to_le_bytes());
        state[0x14..0x18].copy_from_slice(&rsrc1.to_le_bytes());
        state[0x18..0x1c].copy_from_slice(&rsrc2.to_le_bytes());
        state
    }

    /// Kind 7 long enough for the code-size word at `+0x08` and no records.
    fn records(code_bytes: usize) -> [u8; 0x18] {
        let mut meta = [0u8; 0x18];
        let dwords = u32::try_from(code_bytes / 4).unwrap();
        meta[8..12].copy_from_slice(&dwords.to_le_bytes());
        meta
    }

    #[test]
    fn parses_a_synthetic_fspx_blob() {
        let kind0 = dispatch_state([8, 16, 1], true, 0x11, 0x22);
        let kind1 = [1u8; 20];
        let kind6 = [0x55u8; 64];
        let kind7 = records(kind6.len());
        let blob = Fspx::assemble(&kind0, &kind1, &kind6, &kind7).unwrap();
        let parsed = Fspx::parse(&blob).unwrap();
        assert_eq!(parsed.dispatch_state, Some(kind0.as_slice()));
        assert_eq!(parsed.kind1, Some(kind1.as_slice()));
        assert_eq!(parsed.program, Some(kind6.as_slice()));
        assert_eq!(parsed.records, Some(kind7.as_slice()));
        let info = parsed.dispatch_info().unwrap();
        assert_eq!(info.threads, [8, 16, 1]);
        assert!(info.wave32);
        assert_eq!((info.rsrc1, info.rsrc2), (0x11, 0x22));
        // The same bytes the translator already accepts.
        let native = crate::native::fspx_program(&blob).unwrap();
        assert_eq!(native.code, kind6);
        assert!(!native.wave64);
        match native.io {
            crate::native::StageIo::Cs(cs) => {
                assert_eq!(cs.threads, info.threads);
                assert_eq!(cs.rsrc2, info.rsrc2);
            }
            _ => panic!("compute program"),
        }
        let dxbc = build(&[(FourCC::ACCC, &blob)]);
        let wrapped = inspect_fspx(&dxbc).unwrap();
        assert_eq!(wrapped.program, Some(kind6.as_slice()));
        assert_eq!(inspect_fspx(&blob).unwrap().program, Some(kind6.as_slice()));
        let mut short = blob.clone();
        short.pop();
        assert!(Fspx::parse(&short).is_err());
        assert!(inspect_fspx(&build(&[(FourCC::ACCC, b"nope")])).is_none());
        assert!(inspect_fspx(b"DXBC").is_none());
        let bare = Fspx::assemble(&[0; 8], &[], &[], &[]).unwrap();
        assert!(Fspx::parse(&bare).unwrap().dispatch_info().is_none());
    }
}
