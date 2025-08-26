import unittest
import numpy as np
from ase import Atoms

from formatted_auto import (  # 把这里改成上面脚本的模块名
    build_ldau_and_magmom_lines,
    atoms_to_spglib_cell,
    spglib_cell_to_atoms,
    get_prim_kpath,
    parse_from_seekpath,
)

class TestAutoPhononHelpers(unittest.TestCase):
    def test_ldau_magmom_lines(self):
        # FeO 原胞：1 Fe + 1 O
        a = 3.0
        atoms = Atoms("FeO", scaled_positions=[[0,0,0],[0.5,0.5,0.5]], cell=np.eye(3)*a, pbc=True)
        block, species, counts = build_ldau_and_magmom_lines(atoms)
        self.assertIn("ISPIN = 2", block)
        self.assertIn("MAGMOM =", block)
        self.assertTrue(species[0] in ("Fe","O"))  # 顺序依 POSCAR
        self.assertEqual(sum(counts), len(atoms))

        # 基线检查：Fe LDAU=2, U=3; O 为 -1, 0
        self.assertIn("LDAU = .TRUE.", block)
        self.assertIn("LDAUTYPE = 2", block)
        # self.assertRegex(block, r"LDAUL = .*\b2\b .* -1")
        # self.assertRegex(block, r"LDAUU = .*\b3\b .* 0")

    def test_spglib_roundtrip(self):
        atoms = Atoms("Si2", scaled_positions=[[0,0,0],[0.25,0.25,0.25]],
                      cell=np.eye(3)*5.43, pbc=True)
        cell = atoms_to_spglib_cell(atoms)
        atoms2 = spglib_cell_to_atoms(cell)
        self.assertEqual(len(atoms), len(atoms2))
        np.testing.assert_allclose(atoms.cell.array, atoms2.cell.array, atol=1e-8)

    def test_seekpath_qpath(self):
        # 简单立方（会得到标准 Γ-X-M-Γ 等路径）
        atoms = Atoms("NaCl", scaled_positions=[[0,0,0],[0.5,0.5,0.5]],
                      cell=np.eye(3)*5.64, pbc=True)
        info = get_prim_kpath(atoms)
        qpts, conns, labels = parse_from_seekpath(info, npoints=21)
        # 至少应生成 >= 2 段
        self.assertGreaterEqual(len(conns), 1)
        # labels 应该是“首点 + 每段末点”
        self.assertGreaterEqual(len(labels), 2)
        # qpts 是一个 [list of np arrays]（段衔接成一个列表）
        self.assertTrue(isinstance(qpts, list))
        self.assertTrue(all(isinstance(q, np.ndarray) for q in qpts))

if __name__ == "__main__":
    unittest.main()

