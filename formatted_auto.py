from __future__ import annotations

import gzip
import os
import sys
import shutil
import json
import time
import yaml
import spglib
import seekpath
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional, Union
from itertools import chain

import numpy as np
from ase import Atoms
from ase.io import read, write

from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.file_IO import write_FORCE_CONSTANTS
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

from pymatgen.io.vasp.outputs import Vasprun

# ==== orchestrator ====
from orchestrator.machine import MachineData
from orchestrator.orchestrator import Orchestrator
from orchestrator.jobs import Job


# ---------------------------
# 用户需要按环境修改的“少数”参数
# ---------------------------

MACHINE_KWARGS = {
    "name": "mp-test",
    "executor": "local",
    "scheduler": "slurm",
    "numb_node": 1,
    "numb_cpu_per_node": 48,
    "machine_capacity": 5,
    "group_size": 1,
    "username": "USER",
    "host": "remote.host",
    "_key_filename": "/home/USER/.ssh/id_rsa",
    "password": None,
    "port": 22,
    "envs": [
        "ulimit -s unlimited",
        "source /public/env/compiler_oneapi2024",
        "# source /public/env/compiler_intel2025",
        "export I_MPI_ADJUST_REDUCE=3",
        "export LD_PRELOAD=/public/home/fanchunpku/qinchuan/soft/vasp-mod/libisintel.so",
        "export I_MPI_FABRICS=shm",
        "export I_MPI_SHM=skx_avx512",
        "export UCX_TLS=sm,dc",
        "export UCX_IB_PCI_RELAXED_ORDERING=on",
    ],
    "queue": "amd9654",
}

# 假 VASP 命令（本地演示用）；生产环境改为你的 mpirun/srun 调用
VASP_COMMAND = 'ulimit -s unlimited; touch OUTCAR; touch CONTCAR; touch vasprun.xml; touch vasp.log; sleep 5'
VASP_COMMAND = "mpirun -np 48 /public/home/lvjian/vasp.6.4.3/vasp.6.4.3/bin/vasp_std > vasp.log 2>&1"

# 本地与远端根
LOCAL_ROOT = Path("./work_dir").expanduser()
REMOTE_ROOT = None

# === POTCAR ===
CONF_FILE = Path("~/.auto_phono.yaml").expanduser()
if not CONF_FILE.exists():
    raise FileNotFoundError(f"缺少 POTCAR 配置文件: {CONF_FILE}")

POTCAR_CONF = yaml.safe_load(open(CONF_FILE, "r"))
# 允许配置里提供一个 family（例如 "POT_GGA_PAW_PBE"），否则默认同名目录
family = POTCAR_CONF.get("POTCAR_FAMILY", "POT_GGA_PAW_PBE")
POTCAR_DIR = Path(POTCAR_CONF["VASP_PSP_DIR"]).expanduser() / family

MAP_FILE = Path(POTCAR_CONF["POTCAR_MAPPING"]).expanduser()
_map_loaded = yaml.safe_load(open(MAP_FILE, "r"))
# 兼容两种格式：{"POTCAR": {...}} 或直接 {...}
POTCAR_MAP: Dict[str, str] = _map_loaded.get("POTCAR", _map_loaded)

# Phonopy 设置
SUPERCELL_MATRIX = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
DISPLACEMENT_DISTANCE = 0.01
SYMPREC = 1e-5

WATCH_INTERVAL = 60
STATE_FILE = (LOCAL_ROOT / ".auto_phonon_state.json")
CANDIDATE_PATTERNS = ("POSCAR", "PRIMCELL.vasp", "*.vasp", "*.cif")

# VASP 输入模板
INCAR_TEMPLATE = """\
ISTART = 0
ICHARG = 2
ADDGRID = True
ALGO = Normal
EDIFF = 1e-08
EDIFFG = -0.0001
ENCUT = 600
GGA = Ps
IBRION = 2
ISIF = 3
ISMEAR = 0
ISYM = 2
KSPACING = 0.2
LASPH = True
LCHARG = False
LMAXMIX = 6
LORBIT = 11
LREAL = False
LWAVE = False
NCORE = 12
NELM = 150
NPAR = 8
NSW = 0
PREC = Accurate
PSTRESS = 0
SIGMA = 0.05
SYSTEM = Magcalc
# ISPIN = 2
# MAGMOM = 54*5.0 216*0.0
# LDAU = True
# LDAUTYPE = 2
# LDAUL = 2 -1 -1
# LDAUU = 3 0 0
# LDAUJ = 0 0 0
# LDAUPRINT = 1
"""

LDAUU_MAP = {"Sc": 3, "Ti": 3, "V": 3, "Cr": 3, "Mn": 3, "Fe": 3, "Co": 3, "Ni": 3, "O": 0, "F": 0, "S": 0, "Cl": 0, "Br": 0, "I": 0, "N": 0, "P": 0, "As": 0, "Tl": 0, "Ge": 0, "Pb": 0, "Sb": 0, "Bi": 0, "Se": 0, "Te": 0, "Po": 0, "Sn": 0}
LDAUL_MAP = {"Sc": 2, "Ti": 2, "V": 2, "Cr": 2, "Mn": 2, "Fe": 2, "Co": 2, "Ni": 2, "O": -1, "F": -1, "S": -1, "Cl": -1, "Br": -1, "I": -1, "N": -1, "P": -1, "As": -1, "Tl": -1, "Ge": -1, "Pb": -1, "Sb": -1, "Bi": -1, "Se": -1, "Te": -1, "Po": -1, "Sn": -1}
MAGMOM_MAP = {"Sc": 5, "Ti": 5, "V": 5, "Cr": 5, "Mn": 5, "Fe": 5, "Co": 5, "Ni": 5, "F": 0, "O": 0, "S": 0, "Cl": 0, "Br": 0, "I": 0, "N": 0, "P": 0, "As": 0, "Tl": 0, "Ge": 0, "Pb": 0, "Sb": 0, "Bi": 0, "Se": 0, "Te": 0, "Po": 0, "Sn": 0}

NPOINTS_PER_SEG = 51


# ---------------------------
# 工具函数
# ---------------------------

def ase_to_phonopy_atoms(at: Atoms) -> PhonopyAtoms:
    return PhonopyAtoms(
        symbols=at.get_chemical_symbols(),
        cell=at.cell.array,
        scaled_positions=at.get_scaled_positions(),
    )

def find_structures(root: Path,
                    patterns: Tuple[str, ...] = ("POSCAR", "CONTCAR", "*.vasp", "*.cif")) -> List[Path]:
    found: List[Path] = []
    for pat in patterns:
        found.extend(sorted(root.rglob(pat)))
    uniq = sorted(set(p for p in found if p.is_file()))
    return uniq

def ensure_clean_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def read_gz_file(file_path: Path) -> str:
    with gzip.open(file_path, 'rt') as f:
        return f.read()

def assemble_potcar(elements: Iterable[str], libdir: Path, mapping: Dict[str, str], outpath: Path):
    """按元素顺序拼接 POTCAR"""
    potcar_text = ""
    for el in elements:
        if el not in mapping:
            raise FileNotFoundError(f"POTCAR_MAP 缺少元素 {el} 的条目")
        ele_pot = Path(mapping[el])
        cand = libdir / ele_pot
        print(cand)
        if cand.is_dir():
            potcar_file = cand / "POTCAR"
        else:
            potcar_file = libdir / Path("POTCAR." + str(ele_pot))
        if not potcar_file.exists():
            gz = Path(str(potcar_file) + ".gz")
            if gz.exists():
                potcar_text += read_gz_file(gz)
            else:
                raise FileNotFoundError(f"未找到 {el} 的 POTCAR 文件：{potcar_file} 或 {gz}")
        else:
            potcar_text += potcar_file.read_text()
    outpath.write_text(potcar_text)

def phonopy_atoms_to_ase_atoms(pha: PhonopyAtoms) -> Atoms:
    return Atoms(
        symbols=pha.symbols,
        cell=pha.cell,
        scaled_positions=pha.scaled_positions,
        pbc=[True, True, True],
    )

def write_poscar_from_phonopy_atoms(pha: PhonopyAtoms, outpath: Path):
    at = phonopy_atoms_to_ase_atoms(pha)
    write(outpath.as_posix(), at, format="vasp", vasp5=True, direct=True)

def parse_forces_from_vasprun(vxml: Path) -> np.ndarray:
    vr = Vasprun(vxml.as_posix(), parse_eigen=False, parse_dos=False, parse_projected_eigen=False)
    return np.array(vr.ionic_steps[-1]["forces"])

def atoms_to_spglib_cell(atoms: Atoms) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ASE Atoms -> spglib cell tuple"""
    return (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers)

def spglib_cell_to_atoms(cell: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Atoms:
    """spglib cell tuple -> ASE Atoms"""
    return Atoms(numbers=cell[2], cell=cell[0], scaled_positions=cell[1], pbc=[True, True, True])

def find_prim_atoms(atoms: Atoms, symprec: float) -> Atoms:
    cell = atoms_to_spglib_cell(atoms)
    prim = spglib.find_primitive(cell, symprec=symprec)
    if prim is None:
        # 找不到原胞就退回原结构
        return atoms.copy()
    return spglib_cell_to_atoms(prim)

def get_prim_kpath(struct: Union[Atoms, Tuple], symprec: float = 1e-5, angle_tolerance: float = -1.0) -> dict:
    """使用 seekpath 获取标准高对称路径（基于原胞）"""
    if isinstance(struct, Atoms):
        struct = atoms_to_spglib_cell(struct)
    path_info = seekpath.get_path(struct, with_time_reversal=True, symprec=symprec, angle_tolerance=angle_tolerance)
    return path_info

def parse_band_path_and_labels(band_str: str, labels: str, npoints: int):
    """保留旧接口：输入逗号分隔字符串的路径 + 空格分隔 label"""
    parts = [p.strip() for p in band_str.split(",")]
    path = [list(map(lambda x: float(eval(x)), seg.split())) for seg in parts]
    qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=npoints)
    label_list = labels.split()
    return qpoints, connections, label_list

def parse_from_seekpath(path_info: dict, npoints: int):
    """
    基于 seekpath 的结果构造 phonopy 需要的 qpoints / connections / labels
    - path_info["point_coords"]: {label: [kx, ky, kz]}
    - path_info["path"]: [ (label_i, label_j), ... ]
    """
    pts = path_info["point_coords"]
    segs: List[Tuple[str, str]] = path_info["path"]
    # segments: [[p_i, p_j], ...]
    path = [[pts[a], pts[b]] for (a, b) in segs]
    qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=npoints)
    # labels：节点序列（首点 + 每段末点）
    label_seq: List[str] = [segs[0][0]] + [b for (_, b) in segs]
    return qpoints, connections, label_seq


# ---------------------------
# LDAU & MAGMOM 组装
# ---------------------------

def build_ldau_and_magmom_lines(
    atoms: Atoms,
    magmom_map: Dict[str, float] = MAGMOM_MAP,
    ldaul_map: Dict[str, int] = LDAUL_MAP,
    ldauu_map: Dict[str, float] = LDAUU_MAP,
    ldauj_map: Dict[str, float] | None = None,
    force_ispin2_if_any_mag: bool = True,
) -> Tuple[str, List[str], List[int]]:
    symbols = atoms.get_chemical_symbols()

    species_order: List[str] = []
    for s in symbols:
        if s not in species_order:
            species_order.append(s)

    counts: List[int] = [symbols.count(sp) for sp in species_order]

    ldaul_vals: List[int] = [int(ldaul_map.get(sp, -1)) for sp in species_order]
    ldauu_vals: List[float] = [float(ldauu_map.get(sp, 0.0)) for sp in species_order]
    ldauj_vals: List[float] = [float((ldauj_map or {}).get(sp, 0.0)) for sp in species_order]

    magmom_groups: List[str] = []
    any_mag = False
    for sp, n in zip(species_order, counts):
        m = float(magmom_map.get(sp, 0.0))
        if abs(m) > 1e-12:
            any_mag = True
        magmom_groups.append(f"{n}*{m:.1f}")
    magmom_line = "MAGMOM = " + " ".join(magmom_groups)

    need_ldau = (any(u > 1e-12 for u in ldauu_vals)) or (any(l >= 0 for l in ldaul_vals))
    need_spin = any_mag if force_ispin2_if_any_mag else False

    lines: List[str] = []
    if need_spin:
        lines.append("ISPIN = 2")
        lines.append(magmom_line)

    if need_ldau:
        lines.append("LDAU = .TRUE.")
        lines.append("LDAUTYPE = 2")
        lines.append("LDAUL = " + " ".join(str(v) for v in ldaul_vals))
        lines.append("LDAUU = " + " ".join(f"{v:g}" for v in ldauu_vals))
        lines.append("LDAUJ = " + " ".join(f"{v:g}" for v in ldauj_vals))
        lines.append("LDAUPRINT = 1")

    incar_block = "\n".join(lines) + ("\n" if lines else "")
    return incar_block, species_order, counts


# ---------------------------
# 核心工作流
# ---------------------------

def build_disp_jobs_for_one_structure(struct_path: Path,
                                      local_root: Path,
                                      remote_root: Path,
                                      supercell: List[List[int]],
                                      dx: float,
                                      symprec: float) -> Tuple[List[Job], Dict]:
    """为一个结构生成位移超胞，写入 VASP 输入，并返回 Job 列表与元数据。"""
    atoms = read(struct_path.as_posix())
    prim_atoms = find_prim_atoms(atoms, symprec=0.1)
    unit = ase_to_phonopy_atoms(prim_atoms)

    ph = Phonopy(unitcell=unit, supercell_matrix=supercell, symprec=symprec)
    ph.generate_displacements(distance=dx)
    supercells = ph.supercells_with_displacements
    if not supercells:
        raise RuntimeError(f"{struct_path} 未生成任何位移超胞，请检查设置")

    struct_dir = struct_path.parent
    phonon_root = struct_dir / "phonon"
    ensure_clean_dir(phonon_root)

    disp_dirs: List[Path] = []
    jobs: List[Job] = []

    elements = atoms.get_chemical_symbols()
    unique_elements_in_order: List[str] = []
    for s in atoms.symbols:
        if s not in unique_elements_in_order:
            unique_elements_in_order.append(s)

    for i, sc in enumerate(supercells, start=1):
        disp_tag = f"disp-{i:03d}"
        ddir = phonon_root / disp_tag
        ensure_clean_dir(ddir)

        _at = phonopy_atoms_to_ase_atoms(sc)

        # POSCAR
        write_poscar_from_phonopy_atoms(sc, ddir / "POSCAR")

        # INCAR + MAGMOM/LDAU 追加
        _incar_block, _species_order, _counts = build_ldau_and_magmom_lines(_at)
        assert set(unique_elements_in_order) == set(_species_order)
        (ddir / "INCAR").write_text(INCAR_TEMPLATE + _incar_block)

        # POTCAR
        assemble_potcar(unique_elements_in_order, POTCAR_DIR, POTCAR_MAP, ddir / "POTCAR")

        # Job
        rel_work_dir = str(ddir.relative_to(local_root))
        job = Job(
            name=f"{struct_dir.name}-{disp_tag}",
            command=VASP_COMMAND,
            local_root=str(local_root),
            local_work_dir=rel_work_dir,
            remote_root=None,
            remote_work_dir=None,
            upload_files=["INCAR", "POSCAR", "POTCAR"],
            download_files=["vasprun.xml", "OUTCAR", "CONTCAR", "vasp.log"],
            # download_current_folders=True,
        )
        jobs.append(job)
        disp_dirs.append(ddir)

    meta = {
        "phonopy": ph,
        "disp_dirs": disp_dirs,
        "phonon_root": phonon_root,
        "struct_dir": struct_dir,
        "struct_path": struct_path,
        "prim": prim_atoms,
    }
    return jobs, meta


def collect_forces_and_build_fc(meta: Dict, npoints: int):
    """在所有位移任务完成后，收集 force，计算 FC，输出文件，并基于 seekpath 自动生成能带。"""
    ph: Phonopy = meta["phonopy"]
    disp_dirs: List[Path] = meta["disp_dirs"]
    phonon_root: Path = meta["phonon_root"]

    set_of_forces: List[np.ndarray] = []
    for ddir in disp_dirs:
        vxml = ddir / "vasprun.xml"
        if not vxml.exists():
            raise FileNotFoundError(f"缺少 {vxml}，请检查 VASP 任务是否完成/回传")
        forces = parse_forces_from_vasprun(vxml)
        set_of_forces.append(forces)

    ph.forces = set_of_forces
    ph.produce_force_constants()

    # 保存 FC 与参数
    write_FORCE_CONSTANTS(ph.force_constants)
    shutil.copyfile("FORCE_CONSTANTS", (phonon_root / "FORCE_CONSTANTS").as_posix())
    ph.save(filename=str(phonon_root / "phonopy_params.yaml"), settings={"force_constants": True})

    # === 自动生成能带路径（seekpath，基于原胞） ===
    path_info = get_prim_kpath(meta["prim"])
    qpoints, connections, label_seq = parse_from_seekpath(path_info, npoints=npoints)

    ph.run_band_structure(qpoints, path_connections=connections, labels=label_seq, with_eigenvectors=True)
    ph.write_yaml_band_structure(filename=str(phonon_root / "band.yaml"))


# ---------------------------
# 目录监听 & 调度
# ---------------------------

def _file_sig(p: Path) -> str:
    h = hashlib.sha1()
    b = p.read_bytes()
    h.update(b)
    st = p.stat()
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()

def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}

def _save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))

def _discover(local_root: Path) -> List[Path]:
    return find_structures(local_root, patterns=CANDIDATE_PATTERNS)

def process_one_structure(struct: Path):
    print(f"[Run] 开始处理结构: {struct.relative_to(LOCAL_ROOT)}")
    jobs, meta = build_disp_jobs_for_one_structure(
        struct_path=struct,
        local_root=LOCAL_ROOT,
        remote_root=REMOTE_ROOT,
        supercell=SUPERCELL_MATRIX,
        dx=DISPLACEMENT_DISTANCE,
        symprec=SYMPREC,
    )
    print(f"[Info] 生成 {len(jobs)} 个位移子任务 -> {meta['phonon_root'].relative_to(LOCAL_ROOT)}")

    machine_data = MachineData.from_kwargs(**MACHINE_KWARGS)
    with Orchestrator(
        machinelist=[asdict(machine_data)],
        timeinterval=30,
        pickup=False,
    ) as orch:
        orch.orchestrate(jobs, nowait=False)

    print("[Info] 远端 VASP 计算完成，开始收集力并构建 FORCE_CONSTANTS / band ...")
    collect_forces_and_build_fc(meta=meta, npoints=NPOINTS_PER_SEG)
    print(f"[OK] {meta['struct_dir'].name} 声子完成 -> {meta['phonon_root']}")

def watch_main():
    local_root = LOCAL_ROOT.resolve()
    if not local_root.exists():
        print(f"[Error] LOCAL_ROOT 不存在：{local_root}")
        sys.exit(1)

    state = _load_state()
    print(f"[Info] 进入监听模式：根目录 = {local_root}")
    print(f"[Info] 已记录 {len(state)} 个历史结构签名；每 {WATCH_INTERVAL}s 扫描一次。Ctrl+C 退出。")

    try:
        while True:
            candidates = _discover(local_root)
            todo: List[Path] = []
            for p in candidates:
                sig = _file_sig(p)
                key = str(p.resolve())
                prev = state.get(key, {})
                if prev.get("sig") != sig:
                    todo.append(p)

            if todo:
                print(f"[Info] 发现 {len(todo)} 个新/变更结构：")
                for tp in todo:
                    print("  -", tp.relative_to(local_root))

                for tp in todo:
                    try:
                        process_one_structure(tp)
                        state[str(tp.resolve())] = {
                            "sig": _file_sig(tp),
                            "t_done": int(time.time())
                        }
                        _save_state(state)
                    except Exception as e:
                        print(f"[Warn] 处理失败 {tp}: {e}")
                        state[str(tp.resolve())] = {
                            "sig": _file_sig(tp),
                            "t_failed": int(time.time()),
                            "error": str(e)
                        }
                        _save_state(state)
            else:
                print("[Info] 无新增变更结构。")

            time.sleep(WATCH_INTERVAL)
    except KeyboardInterrupt:
        print("\n[Exit] 已退出监听。")

if __name__ == "__main__":
    watch_main()

