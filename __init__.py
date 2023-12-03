import os
import subprocess
import sys
import numpy as np


def _dummyimport():
    import Cython


try:
    from .numpymap import maparray, mapstringarray

except Exception as e:
    cstring = r"""# distutils: language=c
# distutils: extra_compile_args=/openmp
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False


from cython.parallel cimport prange
cimport cython
import numpy as np
cimport numpy as np
import cython

ctypedef fused alldtypes:
    cython.bint
    cython.char
    cython.schar
    cython.uchar
    cython.short
    cython.ushort
    cython.int
    cython.uint
    cython.long
    cython.ulong
    cython.longlong
    cython.ulonglong
    cython.float
    cython.double
    cython.longdouble
    cython.floatcomplex
    cython.doublecomplex
    cython.longdoublecomplex
    cython.size_t
    cython.Py_ssize_t
    cython.Py_hash_t
    cython.Py_UCS4
ctypedef fused alldtypes2:
    cython.bint
    cython.char
    cython.schar
    cython.uchar
    cython.short
    cython.ushort
    cython.int
    cython.uint
    cython.long
    cython.ulong
    cython.longlong
    cython.ulonglong
    cython.float
    cython.double
    cython.longdouble
    cython.floatcomplex
    cython.doublecomplex
    cython.longdoublecomplex
    cython.size_t
    cython.Py_ssize_t
    cython.Py_hash_t
    cython.Py_UCS4
cpdef void maparray(alldtypes2[:] dstp,alldtypes[:] srcp,alldtypes[:] keys ,alldtypes2[:] values  ):
    cdef Py_ssize_t k,kk
    cdef Py_ssize_t len_dstp= len(dstp)
    cdef Py_ssize_t len_keys= len(keys)

    for k in prange(len_dstp,nogil=True):
        for kk in range(len_keys):
            if keys[kk]==srcp[k]:
                dstp[k]=values[kk]
                break

cpdef void mapstringarray(alldtypes[:] ar, alldtypes[:] keysnp, cython.uchar[:,:] keysv,cython.uchar[:] out, Py_ssize_t itemsize):
    cdef Py_ssize_t lenelement=len(ar)
    cdef Py_ssize_t lenkeysnp=len(keysnp)
    cdef bint isele
    cdef Py_ssize_t element,i,letterc
    for element in prange(lenelement,nogil=True):
        for i in range(lenkeysnp):
            isele=keysnp[i]==ar[element]
            if isele:
                for letterc in range(itemsize):
                    out[element*itemsize+letterc]=keysv[i][letterc]
                break



"""
    pyxfile = f"numpymap.pyx"
    pyxfilesetup = f"numpymapcompiled_setup.py"

    dirname = os.path.abspath(os.path.dirname(__file__))
    pyxfile_complete_path = os.path.join(dirname, pyxfile)
    pyxfile_setup_complete_path = os.path.join(dirname, pyxfilesetup)

    if os.path.exists(pyxfile_complete_path):
        os.remove(pyxfile_complete_path)
    if os.path.exists(pyxfile_setup_complete_path):
        os.remove(pyxfile_setup_complete_path)
    with open(pyxfile_complete_path, mode="w", encoding="utf-8") as f:
        f.write(cstring)
    numpyincludefolder = np.get_include()
    compilefile = (
            """
	from setuptools import Extension, setup
	from Cython.Build import cythonize
	ext_modules = Extension(**{'py_limited_api': False, 'name': 'numpymap', 'sources': ['numpymap.pyx'], 'include_dirs': [\'"""
            + numpyincludefolder
            + """\'], 'define_macros': [], 'undef_macros': [], 'library_dirs': [], 'libraries': [], 'runtime_library_dirs': [], 'extra_objects': [], 'extra_compile_args': [], 'extra_link_args': [], 'export_symbols': [], 'swig_opts': [], 'depends': [], 'language': None, 'optional': None})

	setup(
		name='numpymap',
		ext_modules=cythonize(ext_modules),
	)
			"""
    )
    with open(pyxfile_setup_complete_path, mode="w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [x.lstrip().replace(os.sep, "/") for x in compilefile.splitlines()]
            )
        )
    subprocess.run(
        [sys.executable, pyxfile_setup_complete_path, "build_ext", "--inplace"],
        cwd=dirname,
        shell=True,
        env=os.environ.copy(),
    )
    try:
        from .numpymap import maparray, mapstringarray

    except Exception as fe:
        sys.stderr.write(f'{fe}')
        sys.stderr.flush()


def get_pointer_array(original):
    dty = np.ctypeslib.as_ctypes_type(original.dtype)

    b = original.ctypes.data
    buff = (dty * original.size).from_address(b)

    aflat = np.frombuffer(buff, dtype=original.dtype)
    return aflat

def map_numpy_array(src,mapdict,keepnotmapped=True,dtype=None):
    values=np.array(list(mapdict.values()))

    if not dtype:
        dtype=values.dtype
    if keepnotmapped:
        dst = np.ascontiguousarray(src.astype(dtype))
    else:
        dst=np.zeros(src.shape,dtype=dtype)
    keys=np.array(list(mapdict.keys()),dtype=src.dtype)

    srcp=get_pointer_array(src)
    dstp=get_pointer_array(dst)
    maparray(dstp,srcp,keys,values)
    return dst

def map_array_with_strings(src,mapdict):

    keysnp = np.array(list(mapdict.keys()))
    keysvx = np.array(list(mapdict.values()))
    arx=np.ascontiguousarray(src)
    isu=False
    if isinstance(keysvx[0],str):
        isu=True
        keysvf=np.ascontiguousarray(np.char.encode(keysvx))

    outputarraystring=np.zeros(arx.shape,dtype=keysvf.dtype)
    keysv=np.ascontiguousarray(keysvf.view(np.uint8).reshape((-1, keysvf.dtype.itemsize)))

    outputarray = (np.zeros_like(outputarraystring))
    outputarrayint=np.ascontiguousarray(outputarray.view(np.uint8))

    ar=get_pointer_array(arx)
    itemsize=int(keysvf.dtype.itemsize)
    out=get_pointer_array(outputarrayint)
    mapstringarray(ar,keysnp,keysv,out,itemsize)
    if isu:
        outputarrayx=np.ascontiguousarray(np.char.decode(outputarray))
        return outputarrayx
    return outputarray
