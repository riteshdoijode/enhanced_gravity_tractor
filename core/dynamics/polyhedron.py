"""Constant-density polyhedron gravity (Werner & Scheeres 1997, Celest. Mech. Dyn. Astron. 65:313).

For a closed polyhedron with outward-oriented faces, density sigma, and field point x:

    U(x)      =  1/2 G sigma sum_e r_e^T E_e r_e L_e  -  1/2 G sigma sum_f r_f^T F_f r_f w_f
    grad U(x) = -G sigma sum_e E_e r_e L_e            +  G sigma sum_f F_f r_f w_f
    lap U(x)  = -G sigma sum_f w_f        (sum_f w_f = 4 pi inside, 0 outside)

r_e, r_f: vectors from x to any point of edge e / face f. F_f = n_f n_f^T. E_e = n_A n_A,e^T + n_B n_B,e^T
(face normals times the in-plane outward edge normals of the two faces sharing e).
L_e = ln((a + b + e)/(a + b - e)) with a, b the distances to the edge ends and e its length.
w_f = signed solid angle of face f seen from x. U > 0 here and the acceleration is +grad U.
Coordinates are Bennu body-fixed (meters, origin at the center of mass).
"""

import numpy as np

from core.constants import BENNU, G


class Polyhedron:
    def __init__(self, vertices, faces, gm=BENNU["gm"]):
        V = np.asarray(vertices, float)
        F = np.asarray(faces, int)
        self.V, self.F = V, F
        p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        n = np.cross(p1 - p0, p2 - p0)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        self.volume = float(np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6.0)
        if self.volume < 0:
            raise ValueError("faces must be oriented with outward normals")
        self.gm = gm
        self.density = gm / (G * self.volume)  # density chosen so total mass matches the measured GM
        self.k = G * self.density
        self.F_dyad = np.einsum("fi,fj->fij", n, n)

        # half-edges (i -> j) in each face's counter-clockwise order; each edge appears twice
        he_i = F.reshape(-1)
        he_j = F[:, [1, 2, 0]].reshape(-1)
        face_of = np.repeat(np.arange(len(F)), 3)
        d = V[he_j] - V[he_i]
        edge_normal = np.cross(d, n[face_of])
        edge_normal /= np.linalg.norm(edge_normal, axis=1, keepdims=True)
        dyad = np.einsum("hi,hj->hij", n[face_of], edge_normal)
        key = np.sort(np.stack([he_i, he_j], axis=1), axis=1)
        uniq, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        if not np.all(counts == 2):
            raise ValueError("mesh is not closed (every edge must be shared by exactly two faces)")
        self.E_dyad = np.zeros((len(uniq), 3, 3))
        np.add.at(self.E_dyad, inv.ravel(), dyad)
        self.edges = uniq  # vertex index pairs
        self.edge_len = np.linalg.norm(V[uniq[:, 1]] - V[uniq[:, 0]], axis=1)

    def _terms(self, x):
        """Per-point edge/face quantities; x shape (M, 3)."""
        V = self.V
        r_v = V[None, :, :] - x[:, None, :]  # (M, Nv, 3) field point -> vertices
        d_v = np.linalg.norm(r_v, axis=2)
        a, b = d_v[:, self.edges[:, 0]], d_v[:, self.edges[:, 1]]
        L = np.log((a + b + self.edge_len) / (a + b - self.edge_len))
        r_e = r_v[:, self.edges[:, 0], :]
        r1, r2, r3 = (r_v[:, self.F[:, k], :] for k in range(3))
        d1, d2, d3 = (d_v[:, self.F[:, k]] for k in range(3))
        num = np.einsum("mfi,mfi->mf", r1, np.cross(r2, r3))
        den = (d1 * d2 * d3 + d1 * np.einsum("mfi,mfi->mf", r2, r3)
               + d2 * np.einsum("mfi,mfi->mf", r3, r1) + d3 * np.einsum("mfi,mfi->mf", r1, r2))
        w = 2.0 * np.arctan2(num, den)
        return r_e, L, r1, w

    def acceleration(self, x, chunk=64):
        """Gravitational acceleration [m/s^2] at body-fixed point(s) x, shape (3,) or (M, 3)."""
        x = np.asarray(x, float)
        single = x.ndim == 1
        x = np.atleast_2d(x)
        out = np.empty_like(x)
        for s in range(0, len(x), chunk):
            r_e, L, r_f, w = self._terms(x[s:s + chunk])
            Er = np.einsum("eij,mej->mei", self.E_dyad, r_e)
            Fr = np.einsum("fij,mfj->mfi", self.F_dyad, r_f)
            out[s:s + chunk] = self.k * (-(Er * L[..., None]).sum(1) + (Fr * w[..., None]).sum(1))
        return out[0] if single else out

    def potential(self, x, chunk=64):
        x = np.atleast_2d(np.asarray(x, float))
        out = np.empty(len(x))
        for s in range(0, len(x), chunk):
            r_e, L, r_f, w = self._terms(x[s:s + chunk])
            eE = np.einsum("mei,eij,mej->me", r_e, self.E_dyad, r_e)
            fF = np.einsum("mfi,fij,mfj->mf", r_f, self.F_dyad, r_f)
            out[s:s + chunk] = 0.5 * self.k * ((eE * L).sum(1) - (fF * w).sum(1))
        return out

    def laplacian(self, x, chunk=64):
        """-G sigma sum(w_f): -4 pi G sigma inside the body, 0 outside (a built-in self-check)."""
        x = np.atleast_2d(np.asarray(x, float))
        return np.concatenate([-self.k * self._terms(x[s:s + chunk])[3].sum(1) for s in range(0, len(x), chunk)])

    def contains(self, x):
        """True where x is inside the body (solid-angle sum = 4 pi)."""
        return self.laplacian(x) < -2.0 * np.pi * self.k
