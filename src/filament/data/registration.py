"""Image-only near-duplicate screening, not preprocessing for the learned model.

Solar-disk normalization and removal of radial illumination stop common disk
outlines from dominating similarity. Bounded registration is a review aid, not
proof that two observations are identical or that no duplicates remain.
"""
import numpy as np
from scipy import ndimage, signal


def circular_support(size, radius=.8):
    axis = (np.arange(size) - (size - 1) / 2) / ((size - 1) / 2)
    y, x = np.meshgrid(axis, axis, indexing='ij')
    return x * x + y * y <= radius * radius


def disk_structure(gray, size=128):
    gray = np.asarray(gray, dtype=np.float32)
    if gray.ndim != 2 or not np.isfinite(gray).all() or min(gray.shape) < 32:
        raise ValueError('Expected a finite grayscale image at least 32 pixels wide.')
    bright = gray[gray > np.percentile(gray, 20)]
    if not len(bright):
        raise ValueError('Cannot estimate disk in a constant image.')
    foreground = gray > .4 * np.median(bright)
    labels, count = ndimage.label(foreground)
    if not count:
        raise ValueError('No solar disk found.')
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    ys, xs = np.nonzero(labels == areas.argmax())
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    ry, rx = (ys.max() - ys.min()) / 2, (xs.max() - xs.min()) / 2
    if min(ry, rx) < min(gray.shape) * .25 or max(ry, rx) / min(ry, rx) > 1.15:
        raise ValueError('Unreliable solar disk extent; manual review required.')
    radius = (rx + ry) / 2
    axis = (np.arange(size) - (size - 1) / 2) / ((size - 1) / 2)
    yy, xx = np.meshgrid(axis, axis, indexing='ij')
    canonical = ndimage.map_coordinates(gray, [cy + yy * radius, cx + xx * radius],
                                       order=1, mode='constant', cval=0)
    structure = ndimage.gaussian_filter(canonical, .6) - ndimage.gaussian_filter(canonical, 4.)
    radial_bins = np.minimum((np.hypot(yy, xx) * 64).astype(int), 127)
    for band in range(58):
        selected = radial_bins == band
        if selected.any():
            structure[selected] -= np.median(structure[selected])
    support = circular_support(size, .88)
    structure[~support] = 0
    std = structure[support].std()
    if std < 1e-5:
        raise ValueError('Insufficient disk structure for a meaningful comparison.')
    structure /= std
    structure = np.clip(structure, -8, 8)
    return structure.astype(np.float32), {'center_yx': [float(cy), float(cx)],
                                        'radius': float(radius), 'structure_std': float(std)}


def orient(array, orientation):
    if orientation not in range(8):
        raise ValueError('Orientation must be an integer from zero through seven.')
    return np.rot90(array[:, ::-1] if orientation >= 4 else array, orientation % 4).copy()


def descriptor(structure):
    if structure.shape != (128, 128):
        raise ValueError('Descriptor expects 128-square disk structure.')
    small = structure.reshape(64, 2, 64, 2).mean(axis=(1, 3))
    small[~circular_support(64, .8)] = 0
    return (small / max(np.linalg.norm(small), 1e-8)).ravel()


def registered_similarity(reference, moving, *, orientation=0, angles=(-4, -2, 0, 2, 4),
                          scales=(.98, 1., 1.02), max_shift=6):
    if reference.shape != moving.shape or reference.ndim != 2 or reference.shape[0] != reference.shape[1]:
        raise ValueError('Registration expects equally sized square structures.')
    size = reference.shape[0]
    moving = orient(moving, orientation)
    support = circular_support(size, .72)
    ref_values = reference[support].astype(float)
    ref_values -= ref_values.mean()
    ref_norm = np.linalg.norm(ref_values)
    if ref_norm <= 1e-8:
        raise ValueError('Reference has no interior structure.')
    ref_match = reference * circular_support(size, .8)
    best = {'ncc': -1., 'orientation': orientation}
    center = np.array([(size - 1) / 2] * 2)
    for angle in angles:
        rotated = ndimage.rotate(moving, angle, reshape=False, order=1, mode='constant')
        for scale in scales:
            matrix = np.eye(2) / scale
            warped = ndimage.affine_transform(rotated, matrix, offset=center - matrix @ center,
                                              order=1, mode='constant')
            corr = signal.correlate(ref_match, warped * circular_support(size, .8), mode='full', method='fft')
            c = size - 1
            window = corr[c - max_shift:c + max_shift + 1, c - max_shift:c + max_shift + 1]
            # Evaluate the three strongest translation candidates by actual NCC.
            for flat in np.argsort(window.ravel())[-3:]:
                dy, dx = np.unravel_index(flat, window.shape)
                shift = int(dy - max_shift), int(dx - max_shift)
                aligned = ndimage.shift(warped, shift, order=1, mode='constant')
                values = aligned[support].astype(float)
                values -= values.mean()
                ncc = float(ref_values @ values / max(ref_norm * np.linalg.norm(values), 1e-8))
                if ncc > best['ncc']:
                    best = {'ncc': ncc, 'orientation': orientation, 'angle_degrees': angle,
                            'scale': scale, 'shift_yx': list(shift)}
    return best
