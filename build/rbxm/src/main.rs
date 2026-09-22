//! Generates builds/RoNet.rbxm from the repo's source tree, in the exact shape
//! Studio exports: a single root ModuleScript `RoNet` (the DI wiring with
//! `require("./dir/Mod")` rewritten to `require(script.dir.Mod)`) containing
//! one Folder per source directory, each holding one ModuleScript per .luau
//! file (files starting with `_` are skipped).
//!
//!   cargo run --release            # write builds/RoNet.rbxm
//!   cargo run --release -- dump <file.rbxm>   # print an existing rbxm's tree

use rbx_binary;
use rbx_dom_weak::types::Variant;
use rbx_dom_weak::{InstanceBuilder, Ustr, WeakDom};
use std::error::Error;
use std::fs::{self, File};
use std::io::BufReader;
use std::io::BufWriter;
use std::path::Path;
use std::path::PathBuf;

const FOLDERS: &[&str] = &["core", "nn", "models", "loss", "optim", "train", "data"];

/// `require("./core/Util")` -> `require(script.core.Util)`, preserving any
/// trailing `(deps)` factory call.
fn rewrite_requires(src: &str) -> String {
	let needle = "require(\"./";
	let mut out = String::with_capacity(src.len());
	let mut rest = src;
	while let Some(pos) = rest.find(needle) {
		out.push_str(&rest[..pos]);
		let tail = &rest[pos + needle.len()..];
		let end = tail.find("\")").expect("malformed require path");
		let path = &tail[..end];
		let (dir, module) = path.split_once('/').expect("require path: dir/mod");
		out.push_str(&format!("require(script.{dir}.{module})"));
		rest = &tail[end + 2..];
	}
	out.push_str(rest);
	out
}

fn module_scripts(dir: &Path) -> Vec<(String, String)> {
	let mut out = Vec::new();
	let mut entries = fs::read_dir(dir)
		.unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
		.flatten()
		.filter_map(|e| e.file_name().into_string().ok())
		.collect::<Vec<_>>();
	entries.sort();
	for name in entries {
		let Some(stem) = name.strip_suffix(".luau") else { continue };
		if stem.starts_with('_') {
			continue;
		}
		let src = fs::read_to_string(dir.join(&name))
			.unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.join(&name).display()));
		out.push((stem.to_string(), src));
	}
	out
}

fn build_dom(root: &Path) -> WeakDom {
	let root_src = fs::read_to_string(root.join("RoNet.luau"))
		.unwrap_or_else(|e| panic!("cannot read RoNet.luau: {e}"));
	let mut ronet = InstanceBuilder::new("ModuleScript")
		.with_name("RoNet")
		.with_property("Source", rewrite_requires(&root_src));

	for dir in FOLDERS {
		let mut folder = InstanceBuilder::new("Folder").with_name(*dir);
		for (name, src) in module_scripts(&root.join(dir)) {
			folder = folder.with_child(
				InstanceBuilder::new("ModuleScript")
					.with_name(name)
					.with_property("Source", src),
			);
		}
		ronet = ronet.with_child(folder);
	}

	WeakDom::new(ronet)
}

fn print_tree(dom: &WeakDom) {
	for inst in dom.descendants_of(dom.root_ref()) {
		let depth = dom.full_path_of(inst.referent(), "/").matches('/').count();
		println!("{}{} {}", "  ".repeat(depth), inst.class, inst.name);
	}
}

fn main() -> Result<(), Box<dyn Error>> {
	let args: Vec<String> = std::env::args().collect();
	let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");

	if args.get(1).map(String::as_str) == Some("dump") {
		let file = args.get(2).ok_or("usage: rbxm_build dump <file.rbxm>")?;
		let input = BufReader::new(File::open(file)?);
		let dom = rbx_binary::from_reader(input)?;
		print_tree(&dom);
		return Ok(());
	}

	if args.get(1).map(String::as_str) == Some("dump-src") {
		let file = args.get(2).ok_or("usage: rbxm_build dump-src <file.rbxm> <InstancePath>")?;
		let want = args.get(3).ok_or("usage: rbxm_build dump-src <file.rbxm> <InstancePath>")?;
		let input = BufReader::new(File::open(file)?);
		let dom = rbx_binary::from_reader(input)?;
		for inst in dom.descendants_of(dom.root_ref()) {
			let path = dom.full_path_of(inst.referent(), "/");
			if path.strip_suffix(want).is_some() || path == *want {
				if let Some(Variant::String(src)) = inst.properties.get(&Ustr::from("Source")) {
					print!("{src}");
					return Ok(());
				}
			}
		}
		return Err(format!("no module with path ending in {want}").into());
	}

	let dom = build_dom(&repo);
	let out = repo.join("builds").join("RoNet.rbxm");
	let output = BufWriter::new(File::create(&out)?);
	rbx_binary::to_writer(output, &dom, &[dom.root_ref()])?;
	println!("wrote {}", out.display());
	Ok(())
}