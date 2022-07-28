#!/usr/bin/env python3

from typing import Dict, FrozenSet, Set, Tuple

import argparse
import contextlib
import os
import pathlib
import re
import shutil

import attrs


RE_IMPORTS = re.compile(r'% -- begin imports --\n(.*)% -- end imports --\n', flags=re.DOTALL)
RE_SINGLE_IMPORT = re.compile(r'% import (.+) as (\w+)')
RE_EXPORT = re.compile(r'\\label{(.+)} +% export')
RE_REF = re.compile(r'\\ref{(.+\.)?(.+)}')
RE_PDF = re.compile(r'(?:\\includegraphics|\\quickfig).*{(\w+\.pdf)}')
RE_TEX_SUBIMPORTLEVEL = re.compile(r'\\subimportlevel{(.+)}{(.+)}{(.+)}')
RE_TEX_INPUT = re.compile(r'\\input{(.+)}')


@contextlib.contextmanager
def set_directory(path: pathlib.Path):
    origin = pathlib.Path()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


def path_from_to(source: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
    """Get relative path from source to target"""
    return pathlib.Path(os.path.relpath(target.resolve(), source.resolve()))


def find_included_tex_files_single_file(
        file: pathlib.Path,
) -> Set[pathlib.Path]:
    with open(file, 'r') as readfile:
        text = readfile.read()
    paths: Set[pathlib.Path] = set()

    result = RE_TEX_SUBIMPORTLEVEL.findall(text)
    for directory, filename, _ in result:
        paths.add(file.parent / directory / filename)

    for filepath in RE_TEX_INPUT.findall(text):
        paths.add(file.parent / filepath)

    return paths


def find_included_tex_files_recursively(
        files_to_check: Set[pathlib.Path],
        files_found: Set[pathlib.Path],
) -> Set[pathlib.Path]:
    r"""Recusrively all TeX files used by a given set of root TeX files.

    This works by recursively following the graph defined by \subimportlevel
    calls, with root notes at each file in files_to_check.

    Args:
        files_to_check: Set of paths where we start looking for TeX imports.
        files_found: Paths we've already identified as in the graph of imports.

    Reeturns a set of relative paths to imported files.
    """
    new_files_found: Set[pathlib.Path] = set()
    if len(files_to_check ) == 0:
        return files_found
    for file in files_to_check:
        files_found.add(file)
        imports = find_included_tex_files_single_file(file)
        for imp in imports:
            new_files_found.add(imp)
    return find_included_tex_files_recursively(
            files_to_check=new_files_found,
            files_found=files_found,
    )


def find_included_tex_files(file: pathlib.Path) -> Set[pathlib.Path]:
    """Find all TeX files used by a given main TeX file.

    Args:
        file: Starting point, i.e. main TeX file.

    Reeturns a set of relative paths to files imported into the main file.
    """
    return find_included_tex_files_recursively(
            files_to_check=set([file]),
            files_found=set(),
    )


class PrefixPool:
    def __init__(self):
        self.prefixes: Set[str] = set()

    def get(self) -> str:
        import numpy as np
        r = '0'  # Mollify the typechecker
        while 1:
            r = str(int(10**10 * np.random.random()))[0:10]
            if not r in self.prefixes:
                break
        self.prefixes.add(r)
        return r


@attrs.define(frozen=True)
class Export:
    label: str


@attrs.define(frozen=True)
class Module:
    """
    Attributes
        path: Path to this module's source file.
        exports: All referenceable exports from this module.
        prefix: Prefix prepended to all export labels in this module.
        text: The text of the TeX source file, with all exports rewritten with
            the module's prefix.
    """
    path: pathlib.Path
    exports: FrozenSet[Export]
    prefix: str
    text: str
    pdfs: FrozenSet[pathlib.Path]


def file_to_module(
        file: pathlib.Path,
        prefix: str,
) -> Module:
    with open(file, 'r') as infile:
        text = infile.read()
    exports = frozenset(Export(label=l) for l in RE_EXPORT.findall(text))
    pdfs = frozenset(RE_PDF.findall(text))
    
    def replace(matcher) -> str:
        if matcher is None:
            raise ValueError
        return r'\label{' + prefix + matcher.groups()[0] + '} % export'

    new_text = RE_EXPORT.sub(replace, text)
    return Module(
            path=file,
            exports=exports,
            prefix=prefix,
            text=new_text,
            pdfs=pdfs,
    )


def find_imports(text: str) -> Set[Tuple[str, str]]:
    result_imports = RE_IMPORTS.search(text)
    if result_imports is None:
        return set()
    import_text, = result_imports.groups()
    imports: Set[Tuple[str, str]] = set()
    for line in import_text.split('\n')[:-1]:
        result = RE_SINGLE_IMPORT.match(line)
        if result is None:
            raise ValueError
        module_name, alias = result.groups()
        imports.add((module_name, alias))
    return imports


def get_mangled_text(module: Module, modules: Set[Module]) -> str:
    modules_by_relative_path = {
            path_from_to(module.path.parent, m.path): m
            for m in modules
    }

    imported_modules: Dict[str, Module] = {}  # alias -> module
    # Figure out which modules we're importing
    for alias, pth in {alias: pth for pth, alias in find_imports(module.text)}.items():
        imported_modules[alias] = modules_by_relative_path[pathlib.Path(pth + '.tex')]

    def repl(matcher) -> str:
        module_alias, export_name = matcher.groups()
        if module_alias is None:  # This is an internal reference.
            import_from_module = modules_by_relative_path[module.path.relative_to(module.path.parent)]
        else:  # This is an external reference.
            import_from_module = imported_modules[module_alias[:-1]]  # Drop the dot
        exports_by_name = {e.label: e for e in import_from_module.exports}
        return r"\ref{" + import_from_module.prefix + exports_by_name[export_name].label + "}"

    return RE_REF.sub(repl, module.text, count=0)


def build(
        main_path: pathlib.Path,
        build_path: pathlib.Path,
) -> None:
    all_paths = find_included_tex_files(main_path)
    prefix_pool = PrefixPool()
    modules = {file_to_module(p, prefix_pool.get()) for p in all_paths}
    all_texts = {
            module.path: get_mangled_text(module, modules)
            for module in modules
    }
    for path, text in all_texts.items():
        outpath = build_path / path
        print(f"Writing to {outpath}")
        outpath.parent.mkdir(exist_ok=True, parents=True)
        outpath.write_text(text)
    for module in modules:
        for path_pdf_relative_to_module in module.pdfs:
            path_pdf = module.path.parent / path_pdf_relative_to_module
            new_path = build_path / path_pdf
            shutil.copy(path_pdf, new_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a TeX document")
    args = parser.parse_args()
    build_path = pathlib.Path('build')
    build(
            pathlib.Path('main.tex'),
            build_path,
    )
    with set_directory(build_path):
        os.system("pdflatex main.tex")
        os.system("pdflatex main.tex")
