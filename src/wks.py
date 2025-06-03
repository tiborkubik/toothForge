import math
import numpy as np


def wave_signatures(evals,
                    evecs,
                    dim: int,
                    return_energies: bool = False,
                    energies=None
                    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the wave signature for all vertices

    Args:
        dim (int): Dimensionality (energy spectra) of the signature.
        return_energies (bool, optional): If True the function returns a tuple (signature, energies)
                                          otherwise only the signature is returned. Defaults to False.
        energies (arraylike, optional): Energie spectra used for signature computation.
                                        If None the energy is linearly spaced. Defaults to None.

    Note:
        This signature is based on 'The Wave Kernel Signature: A Quantum Mechanical Approach to Shape Analysis'
        by Mathieu Aubry et al (https://vision.informatik.tu-muenchen.de/_media/spezial/bib/aubry-et-al-4dmod11.pdf)

    Returns:
        Returns an array of shape (#vertices, dim) containing the heat signatures of every vertex.
        If return_times is True this function returns a tuple (Signature, timesteps).
    """

    if energies is None:
        emin = math.log(evals[1])
        emax = math.log(evals[-1]) / 1.02
        energies = np.linspace(emin, emax, dim)
    else:
        energies = np.array(energies).flatten()
        assert len(
            energies) == dim, f"Requested featrue dimension and energies array do not match: {dim} and {len(energies)}"

    sigma = 7.0 * (energies[-1] - energies[0]) / dim
    phi2 = np.square(evecs[:, :])
    exp = np.exp(-np.square(energies[None] - np.log(evals[:, None])) / (2.0 * sigma * sigma))
    s = np.sum(phi2[..., None] * exp[None], axis=1)
    energy_trace = np.sum(exp, axis=0)
    s = s / energy_trace[None]

    if return_energies:
        return s, energies
    else:
        return s
