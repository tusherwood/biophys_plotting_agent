"""
vqvae_cache_io.py

Torch-free row selection on a VQ-VAE crop cache: a `.pt` written by torch.save
holding {key: uint8 tensor (N, ...)} plus its `<stem>_metadata.parquet`, where
parquet row i describes tensor row i (the pair compile_experiment.py writes as
concat_vqvae_cache.pt + concat_vqvae_cache_metadata.parquet).

`filter_cache` writes a new pair holding only the rows of a boolean keep mask,
streaming a chunk of rows at a time, so neither torch nor the whole tensor in
memory is needed: torch.save writes an uncompressed zip whose storage blobs are
C-order buffers, so keeping axis-0 rows of a contiguous tensor is keeping byte
ranges of its blob. The output loads with torch.load like any torch.save file.

Adapted from biophys_helpers' browse_pt.py (the stubbed unpickler that reads a
.pt's tensor shapes without torch) and concat_vqvae_caches.py (the aligned zip
writer and row-selecting blob copy). Kept separate from biophys_plot_toolkit so
the plotting path never imports it; only tk.export_processed_cache does.
"""
import contextlib
import io
import pickle
import sys
import types
import zipfile
from pathlib import Path

import numpy as np

_CHUNK_BYTES = 1 << 20   # streaming copy size
_ZIP_ALIGN   = 64        # torch aligns storage blobs to 64-byte boundaries
_BATCH_ROWS  = 8_192     # parquet rows per streamed batch

# Bytes per element for each torch storage class.
_STORAGE_ITEMSIZE = {
    'ByteStorage': 1, 'CharStorage': 1, 'BoolStorage': 1,
    'ShortStorage': 2, 'HalfStorage': 2, 'BFloat16Storage': 2,
    'IntStorage': 4, 'FloatStorage': 4,
    'LongStorage': 8, 'DoubleStorage': 8, 'ComplexFloatStorage': 8,
    'ComplexDoubleStorage': 16,
}

ROW_COLUMN          = 'cache_row'            # parquet row -> tensor row
ROW_COLUMN_COMPILED = 'cache_row_compiled'   # that row's index in the source cache


# ---------------------------------------------------------------------------
# Reading: tensor shapes from the pickle alone
# ---------------------------------------------------------------------------

class _StorageInfo:
    def __init__(self, key: str, storage_class: str, numel: int):
        self.key           = key
        self.storage_class = storage_class
        self.numel         = numel


class _TensorInfo:
    def __init__(self, storage, shape: tuple, stride: tuple, offset: int):
        self.storage = storage
        self.shape   = tuple(int(d) for d in shape)
        self.stride  = tuple(int(s) for s in stride)
        self.offset  = int(offset)

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def contiguous(self) -> bool:
        """C-order buffer? Size-1 axes are skipped, as in torch's own check."""
        expected = 1
        for size, stride in zip(reversed(self.shape), reversed(self.stride)):
            if size == 1:
                continue
            if stride != expected:
                return False
            expected *= size
        return True


class _Opaque:
    """Stand-in for anything the pickle references besides tensors."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Opaque()

    def __setstate__(self, state):
        pass


def _rebuild_tensor(storage, storage_offset, size, stride, *rest):
    return _TensorInfo(storage if isinstance(storage, _StorageInfo) else None,
                       size, stride, storage_offset)


class _StubUnpickler(pickle.Unpickler):
    """Resolves torch's tensor rebuild calls to _TensorInfo; nothing else runs."""

    def find_class(self, module, name):
        if module == 'torch._utils' and name.startswith('_rebuild_tensor'):
            return _rebuild_tensor
        if module == 'torch._utils' and name.startswith('_rebuild_parameter'):
            return lambda data, *a, **k: data
        if module == 'collections' and name == 'OrderedDict':
            return dict
        return type(name, (_Opaque,), {})

    def persistent_load(self, pid):
        # torch writes ('storage', storage_type, key, location, numel)
        storage_class = getattr(pid[1], '__name__', str(pid[1]))
        return _StorageInfo(str(pid[2]), storage_class, int(pid[4]))


def read_pt(path: Path) -> object:
    """The .pt's object graph with every tensor as a _TensorInfo (no data read)."""
    path = Path(path)
    if not zipfile.is_zipfile(str(path)):
        raise ValueError(f'{path.name} is not a zip-format torch.save file')
    with zipfile.ZipFile(str(path)) as zf:
        pkl = [n for n in zf.namelist() if n.endswith('data.pkl')]
        if not pkl:
            raise ValueError(f'{path.name} holds no data.pkl; not a torch.save file')
        raw = zf.read(min(pkl, key=len))
    return _StubUnpickler(io.BytesIO(raw)).load()


# ---------------------------------------------------------------------------
# Writing: a pickle that torch.load turns into real tensors
# ---------------------------------------------------------------------------

def _rebuild_tensor_v2(*args, **kwargs):
    """Never called: exists so pickle can reference torch._utils._rebuild_tensor_v2."""
    raise NotImplementedError


_rebuild_tensor_v2.__module__ = 'torch._utils'
_rebuild_tensor_v2.__qualname__ = _rebuild_tensor_v2.__name__ = '_rebuild_tensor_v2'


@contextlib.contextmanager
def _torch_refs():
    """
    (rebuild function, {storage class name: class}) for the pickle to reference.

    The pickle names torch._utils._rebuild_tensor_v2 and torch.<X>Storage, and
    pickle only emits a global it can import back. If real torch is already
    loaded, its own objects are referenced (nothing is called). Otherwise stub
    modules stand in for the duration of the dump only, so a later real
    `import torch` in the same process is unaffected.
    """
    torch = sys.modules.get('torch')
    if torch is not None and hasattr(torch, '_utils') and \
            hasattr(torch._utils, '_rebuild_tensor_v2'):
        yield torch._utils._rebuild_tensor_v2, {
            n: getattr(torch, n) for n in _STORAGE_ITEMSIZE if hasattr(torch, n)}
        return
    stub = types.ModuleType('torch')
    stub._utils = types.ModuleType('torch._utils')
    stub._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    classes = {}
    for n in _STORAGE_ITEMSIZE:
        cls = type(n, (object,), {})
        cls.__module__, cls.__qualname__ = 'torch', n
        setattr(stub, n, cls)
        classes[n] = cls
    saved = {k: sys.modules.get(k) for k in ('torch', 'torch._utils')}
    sys.modules['torch'], sys.modules['torch._utils'] = stub, stub._utils
    try:
        yield _rebuild_tensor_v2, classes
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class _StorageRef:
    def __init__(self, key: str, storage_class: str, numel: int):
        self.key, self.storage_class, self.numel = key, storage_class, numel


class _TensorPlaceholder:
    """Pickles exactly as a C-contiguous torch tensor of `shape`."""

    def __init__(self, rebuild, storage_key: str, storage_class: str, shape: tuple):
        self.rebuild       = rebuild
        self.storage_key   = storage_key
        self.storage_class = storage_class
        self.shape         = tuple(int(d) for d in shape)

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def __reduce_ex__(self, protocol):
        stride, acc = [], 1
        for d in reversed(self.shape):
            stride.append(acc)
            acc *= d
        return (self.rebuild,
                (_StorageRef(self.storage_key, self.storage_class, self.numel),
                 0, self.shape, tuple(reversed(stride)), False, None))


class _Pickler(pickle.Pickler):
    def __init__(self, fh, classes: dict):
        super().__init__(fh, protocol=2)
        self.classes = classes

    def persistent_id(self, obj):
        if isinstance(obj, _StorageRef):
            return ('storage', self.classes[obj.storage_class], obj.key, 'cpu', obj.numel)
        return None


def _open_aligned(zf: zipfile.ZipFile, name: str, size: int):
    """A STORED zip entry whose payload starts on a 64-byte boundary, as torch writes."""
    info = zipfile.ZipInfo(name)
    info.compress_type = zipfile.ZIP_STORED
    info.file_size = size
    overhead = zf.fp.tell() + 30 + len(name.encode('utf-8'))
    pad = (-overhead) % _ZIP_ALIGN
    if pad:
        info.extra = b'\x00' * pad
    return zf.open(info, mode='w')


def _blob_name(zf: zipfile.ZipFile, storage_key: str) -> str:
    suffix = f'data/{storage_key}'
    for name in zf.namelist():
        if name == suffix or name.endswith(f'/{suffix}'):
            return name
    raise KeyError(f'no storage blob {storage_key!r} in {zf.filename}')


def _read_exact(fh, n: int) -> bytes:
    parts, got = [], 0
    while got < n:
        chunk = fh.read(n - got)
        if not chunk:
            break
        parts.append(chunk)
        got += len(chunk)
    return b''.join(parts)


def _copy_blob_rows(src_path: Path, storage_key: str, dst, expect: int,
                    keep: np.ndarray) -> int:
    """Stream the axis-0 rows of one storage blob where `keep` is True into dst."""
    n = len(keep)
    if n == 0:
        return 0
    if expect % n:
        raise ValueError(f'{src_path.name}: storage {storage_key} is {expect:,} bytes, '
                         f'not a whole number of {n:,} rows')
    row_bytes = expect // n
    rows_per_chunk = max(1, _CHUNK_BYTES // row_bytes)
    written = 0
    with zipfile.ZipFile(str(src_path)) as zf:
        with zf.open(_blob_name(zf, storage_key)) as fh:
            for start in range(0, n, rows_per_chunk):
                stop = min(n, start + rows_per_chunk)
                buf = _read_exact(fh, (stop - start) * row_bytes)
                if len(buf) != (stop - start) * row_bytes:
                    raise ValueError(f'{src_path.name}: storage {storage_key} ended early')
                sel = keep[start:stop]
                if sel.all():
                    dst.write(buf)
                    written += len(buf)
                elif sel.any():
                    rows = np.frombuffer(buf, dtype=np.uint8).reshape(-1, row_bytes)[sel]
                    dst.write(rows.tobytes())
                    written += rows.nbytes
            if fh.read(1):
                raise ValueError(f'{src_path.name}: storage {storage_key} is longer than '
                                 f'expected')
    return written


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

def cache_tensors(pt_path: Path, n_rows: int) -> dict:
    """{key: _TensorInfo} for every top-level tensor; raises if any can't be row-selected."""
    obj = read_pt(pt_path)
    if not isinstance(obj, dict):
        raise ValueError(f'{Path(pt_path).name} holds a {type(obj).__name__}, not a dict')
    tensors = {k: v for k, v in obj.items() if isinstance(v, _TensorInfo)}
    if not tensors:
        raise ValueError(f'{Path(pt_path).name} holds no tensors')
    for key, t in tensors.items():
        if t.storage is None or t.storage.storage_class not in _STORAGE_ITEMSIZE:
            raise ValueError(f'{key}: unknown storage')
        if not t.shape or t.shape[0] != n_rows:
            raise ValueError(f'{key} has shape {t.shape}, but the parquet has {n_rows:,} rows')
        if not t.contiguous or t.offset or t.storage.numel != t.numel:
            raise ValueError(f'{key} is not a contiguous tensor of its own storage')
    return tensors


def filter_cache(pt_in, parquet_in, keep, pt_out, parquet_out) -> dict:
    """
    Write the rows of (pt_in, parquet_in) where `keep` is True to (pt_out, parquet_out).

    Every top-level tensor is kept (non-tensor entries are dropped). In the
    parquet, `cache_row` is renumbered over the kept rows so it still indexes
    the new tensors, and the source row index is kept as `cache_row_compiled`.
    Re-reads both outputs and raises if they disagree. Returns {key: shape}.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    pt_in, parquet_in = Path(pt_in), Path(parquet_in)
    pt_out, parquet_out = Path(pt_out), Path(parquet_out)
    keep = np.asarray(keep, dtype=bool)
    pf = pq.ParquetFile(str(parquet_in))
    n_rows = pf.metadata.num_rows
    if keep.size != n_rows:
        raise ValueError(f'keep mask has {keep.size:,} entries for {n_rows:,} rows')
    tensors = cache_tensors(pt_in, n_rows)
    n_keep = int(keep.sum())
    pt_out.parent.mkdir(parents=True, exist_ok=True)

    # --- tensors ---
    shapes, plan = {}, []
    with _torch_refs() as (rebuild, classes):
        payload = {}
        for i, (key, t) in enumerate(tensors.items()):
            shape = (n_keep,) + t.shape[1:]
            ph = _TensorPlaceholder(rebuild, str(i), t.storage.storage_class, shape)
            payload[key] = ph
            shapes[key] = shape
            item = _STORAGE_ITEMSIZE[t.storage.storage_class]
            plan.append((str(i), t, ph.numel * item, t.numel * item))
        buf = io.BytesIO()
        _Pickler(buf, classes).dump(payload)
    data_pkl = buf.getvalue()

    prefix = 'archive'
    with zipfile.ZipFile(str(pt_out), 'w', zipfile.ZIP_STORED) as zf:
        with _open_aligned(zf, f'{prefix}/data.pkl', len(data_pkl)) as dst:
            dst.write(data_pkl)
        with _open_aligned(zf, f'{prefix}/byteorder', len(sys.byteorder)) as dst:
            dst.write(sys.byteorder.encode('ascii'))
        with _open_aligned(zf, f'{prefix}/version', 2) as dst:
            dst.write(b'3\n')
        for storage_key, t, out_bytes, in_bytes in plan:
            with _open_aligned(zf, f'{prefix}/data/{storage_key}', out_bytes) as dst:
                got = _copy_blob_rows(pt_in, t.storage.key, dst, in_bytes, keep)
            if got != out_bytes:
                raise ValueError(f'wrote {got:,} bytes for storage {storage_key}, '
                                 f'expected {out_bytes:,}')

    # --- parquet ---
    src_schema = pf.schema_arrow
    has_row = ROW_COLUMN in src_schema.names
    fields = [f for f in src_schema if f.name != ROW_COLUMN_COMPILED]
    if has_row:
        fields.append(pa.field(ROW_COLUMN_COMPILED, pa.int64()))
    schema = pa.schema(fields)
    written, offset = 0, 0
    writer = pq.ParquetWriter(str(parquet_out), schema)
    try:
        for batch in pf.iter_batches(batch_size=_BATCH_ROWS):
            sel = keep[offset:offset + batch.num_rows]
            offset += batch.num_rows
            if not sel.any():
                continue
            batch = batch.filter(pa.array(sel))
            n = batch.num_rows
            cols = []
            for f in schema:
                if f.name == ROW_COLUMN:
                    cols.append(pa.array(range(written, written + n), type=f.type))
                elif f.name == ROW_COLUMN_COMPILED:
                    cols.append(batch.column(ROW_COLUMN).cast(pa.int64()))
                else:
                    cols.append(batch.column(f.name))
            writer.write_table(pa.Table.from_arrays(cols, schema=schema))
            written += n
    finally:
        writer.close()
        pf.close()

    # --- self-check: the invariant the pair exists to keep ---
    out_rows = pq.ParquetFile(str(parquet_out)).metadata.num_rows
    out_shapes = {k: v.shape for k, v in cache_tensors(pt_out, out_rows).items()}
    if out_rows != n_keep or out_shapes != shapes:
        raise ValueError(f'written cache is inconsistent: parquet {out_rows:,} rows, '
                         f'tensors {out_shapes}, expected {n_keep:,} rows')
    return shapes
