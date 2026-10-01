"""Shared triangle quadrature rule used by both assembly and verification."""
# 6-point, degree-4 accurate symmetric triangle quadrature.
# Barycentric coordinates; weights sum to 1.
_A1, _B1, _W1 = 0.816847572980459, 0.091576213509771, 0.109951743655322
_A2, _B2, _W2 = 0.108103018168070, 0.445948490915965, 0.223381589678011
TRI_QUAD = [
    (_A1, _B1, _B1, _W1), (_B1, _A1, _B1, _W1), (_B1, _B1, _A1, _W1),
    (_A2, _B2, _B2, _W2), (_B2, _A2, _B2, _W2), (_B2, _B2, _A2, _W2),
]
