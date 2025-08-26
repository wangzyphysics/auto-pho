from pathlib import Path
import numpy as np
from ase import Atoms, io

# 导入你的主脚本模块（把路径和名字替换成实际的）
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import formatted_auto as ap  # 把名字改成你的模块名

# 1) 准备一个简单结构（NaCl）
root = ap.LOCAL_ROOT
root.mkdir(parents=True, exist_ok=True)
case_dir = root / "NaCl_case"
case_dir.mkdir(exist_ok=True)

atoms = Atoms("NaCl", scaled_positions=[[0,0,0], [0.5,0.5,0.5]], cell=np.eye(3)*5.64, pbc=True)
io.write((case_dir / "POSCAR").as_posix(), atoms, format="vasp", vasp5=True, direct=True)

# 2) 直接调用单结构处理（使用本地假 VASP：touch 文件）
ap.process_one_structure(case_dir / "POSCAR")

print("Done. Check:", (case_dir / "phonon").as_posix())

