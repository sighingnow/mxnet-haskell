-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Base.Internal.Raw
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Direct C FFI bindings for <mxnet/c_api.h>.
--
#if __GLASGOW_HASKELL__ >= 709
{-# LANGUAGE Safe #-}
#elif __GLASGOW_HASKELL__ >= 701
{-# LANGUAGE Trustworthy #-}
#endif
{-# LANGUAGE ForeignFunctionInterface #-}

module MXNet.Core.Base.Internal.Raw where

import Foreign.C.Types
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Ptr
import Foreign.Storable

import C2HS.C.Extra.Marshal

{#import MXNet.Core.Internal.Types.Raw #}

#include <mxnet/c_api.h>

-- | Handle size_t type.
{#typedef size_t CSize#}

-- | Get the string message of last error.
{#fun MXGetLastError as mxGetLastError
    {
    } -> `String' #}

-- | Seed the global random number generators in mxnet.
{#fun MXRandomSeed as mxRandomSeed
    { `Int'
    } -> `Int' #}

-- | Notify the engine about a shutdown.
{#fun MXNotifyShutdown as mxNotifyShutdown
    {
    } -> `Int' #}

-- | Create a NDArray handle that is not initialized.
{#fun MXNDArrayCreateNone as mxNDArrayCreateNone
    { alloca- `NDArrayHandle' peek*
    } -> `Int' -- ^ The returned NDArrayHandle.
    #}

-- | Create a NDArray with specified shape.
{#fun MXNDArrayCreate as mxNDArrayCreate
    { withArray* `[MXUInt]'
    , id `MXUInt'
    , `Int'                         -- ^ Device type, specify device we want to take.
    , `Int'                         -- ^ The device id of the specific device.
    , `Int'                         -- ^ Whether to delay allocation until.
    , alloca- `NDArrayHandle' peek*
    } -> `Int' -- ^ The returing handle.
    #}

-- | Create a NDArray with specified shape and data type.
{#fun MXNDArrayCreateEx as mxNDArrayCreateEx
    { withArray* `[MXUInt]'
    , id `MXUInt'
    , `Int'                         -- ^ Device type, specify device we want to take.
    , `Int'                         -- ^ The device id of the specific device.
    , `Int'                         -- ^ Whether to delay allocation until.
    , `Int'                         -- ^ Data type of created array.
    , alloca- `NDArrayHandle' peek*
    } -> `Int' -- ^ The returing handle.
    #}

-- | Create a NDArray handle that is loaded from raw bytes.
{#fun MXNDArrayLoadFromRawBytes as mxNDArrayLoadFromRawBytes
    { id `Ptr ()'                   -- ^ The head of the raw bytes.
    , `CSize'                       -- ^ Size of the raw bytes.
    , alloca- `NDArrayHandle' peek*
    } -> `Int' #}

-- | Save the NDArray into raw bytes.
{#fun MXNDArraySaveRawBytes as mxNDArraySaveRawBytes
    { id `NDArrayHandle'        -- ^ The NDArray handle.
    , alloca- `CSize' peek*     -- ^ Size of the raw bytes.
    , alloca- `Ptr CChar' peek* -- ^ The head of returning memory bytes.
    } -> `Int' #}

-- | Save list of narray into the file.
{#fun MXNDArraySave as mxNDArraySave
    { `String'                      -- ^ Name of the file.
    , id `MXUInt'                   -- ^ Number of arguments to save.
    , withArray* `[NDArrayHandle]'  -- ^ the array of NDArrayHandles to be saved.
    , withStringArray* `[String]'   -- ^ names of the NDArrays to save.
    } -> `Int' #}

{#fun MXNDArrayLoad as mxNDArrayLoadImpl
    { `String'                      -- ^ Name of the file.
    , alloca- `MXUInt' peek*
    , alloca- `Ptr NDArrayHandle' peek*
    , alloca- `MXUInt' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    } -> `Int' #}

-- | Load list of narray from the file.
mxNDArrayLoad :: String                         -- ^ Name of the file.
              -> IO (Int,
                     MXUInt, [NDArrayHandle],
                     MXUInt, [String])          -- ^ The size of ndarray handles, ndarray
                                                -- handles the number of names and the
                                                -- returned names.
mxNDArrayLoad fname = do
    (res, c1, p1, c2, p2) <- mxNDArrayLoadImpl fname
    handles <- peekArray (fromIntegral c1) p1
    names <- peekStringArray c2 p2
    return (res, c1, handles, c2, names)

-- | Perform a synchronize copy from a continugous CPU memory region.
-- This is useful to copy data from existing memory region that are
-- not wrapped by NDArray (thus dependency not being tracked).
{#fun MXNDArraySyncCopyFromCPU as mxNDArraySyncCopyFromCPU
    { id `NDArrayHandle'    -- ^ The NDArrayHandle.
    , id `Ptr ()'           -- ^ The raw data source to copy from.
    , `CSize'               -- ^ The memory size want to copy from.
    } -> `Int' #}

-- | Perform a synchronize copy to a continugous CPU memory region.
{#fun MXNDArraySyncCopyToCPU as mxNDArraySyncCopyToCPU
    { id `NDArrayHandle'    -- ^ The NDArrayHandle.
    , id `Ptr ()'           -- ^ The raw data source to copy into.
    , `CSize'               -- ^ The memory size want to copy into.
    } -> `Int' #}

-- | Wait until all the pending writes with respect NDArray are finished.
{#fun MXNDArrayWaitToRead as mxNDArrayWaitToRead
    { id `NDArrayHandle'
    } -> `Int' #}

-- | Wait until all the pending read/write with respect NDArray are finished.
{#fun MXNDArrayWaitToWrite as mxNDArrayWaitToWrite
    { id `NDArrayHandle'
    } -> `Int' #}

-- | Wait until all delayed operations in the system is completed.
{#fun MXNDArrayWaitAll as mxNDArrayWaitAll
    {
    } -> `Int' #}

-- | Free the narray handle.
{#fun MXNDArrayFree as mxNDArrayFree
    { id `NDArrayHandle'
    } -> `Int' #}

-- | Slice the NDArray along axis 0.
{#fun MXNDArraySlice as mxNDArraySlice
    { id `NDArrayHandle'            -- ^ The handle to the NDArray.
    , id `MXUInt'                   -- ^ The beginning index of slice.
    , id `MXUInt'                   -- ^ The ending index of slice.
    , alloca- `NDArrayHandle' peek*
    } -> `Int' -- ^ The NDArrayHandle of sliced NDArray.
    #}

-- | Index the NDArray along axis 0.
{#fun MXNDArrayAt as mxNDArrayAt
    { id `NDArrayHandle'            -- ^ The handle to the NDArray.
    , id `MXUInt'                   -- ^ The index.
    , alloca- `NDArrayHandle' peek*
    } -> `Int' -- ^ The NDArrayHandle of output NDArray.
    #}

-- | Reshape the NDArray.
{#fun MXNDArrayReshape as mxNDArrayReshape
    { id `NDArrayHandle'            -- ^ The handle to the NDArray.
    , `Int'                         -- ^ Number of dimensions of new shape.
    , alloca- `Int' peekIntegral*
    , alloca- `NDArrayHandle' peek*
    } -> `Int' -- ^ The new shape data and the NDArrayHandle of reshaped NDArray.
    #}

{#fun MXNDArrayGetShape as mxNDArrayGetShapeImpl
    { id `NDArrayHandle'
    , alloca- `MXUInt' peek*
    , alloca- `Ptr MXUInt' peek*
    } -> `Int' #}

-- Get the shape of the array.
mxNDArrayGetShape :: NDArrayHandle
                  -> IO (Int, MXUInt, [MXUInt]) -- ^ The output dimension and it's shape.
mxNDArrayGetShape handle = do
    (res, d, p) <- mxNDArrayGetShapeImpl handle
    shapes <- peekArray (fromIntegral d) p
    return (res, d, shapes)

-- | Get the content of the data in NDArray.
{#fun MXNDArrayGetData as mxNDArrayGetData
    { id `NDArrayHandle'            -- ^ The NDArray handle.
    , alloca- `Ptr MXFloat' peek*
    } -> `Int' -- ^ Pointer holder to get pointer of data.
    #}

-- | Get the type of the data in NDArray
{#fun MXNDArrayGetDType as mxNDArrayGetDType
    { id `NDArrayHandle'            -- ^ The NDArray handle.
    , alloca- `Int' peekIntegral*
    } -> `Int' -- ^ The type of data in this NDArray handle.
    #}

-- | Get the context of the NDArray.
{#fun MXNDArrayGetContext as mxNDArrayGetContext
    { id `NDArrayHandle'          -- ^ The NDArray handle.
    , alloca- `Int' peekIntegral*
    , alloca- `Int' peekIntegral*
    } -> `Int' -- ^ The device type and device id.
    #}

{#fun MXListFunctions as mxListFunctionsImpl
    { alloca- `MXUInt' peek*
    , alloca- `Ptr FunctionHandle' peek*
    } -> `Int' #}

-- | List all the available functions handles.
mxListFunctions :: IO (Int, MXUInt, [FunctionHandle]) -- ^ The output function handle array.
mxListFunctions = do
    (res, c, p) <- mxListFunctionsImpl
    fs <- peekArray (fromIntegral c) p
    return (res, c, fs)

-- | Get the function handle by name.
{#fun MXGetFunction as mxGetFunction
    { `String'                          -- ^ The name of the function.
    , alloca- `FunctionHandle' peek*
    } -> `Int' -- ^ The corresponding function handle.
    #}

{#fun MXFuncGetInfo as mxFuncGetInfoImpl
    { id `FunctionHandle'
    , alloca- `String' peekString*
    , alloca- `String' peekString*
    , alloca- `MXUInt' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    , alloca- `String' peekString*
    } -> `Int' #}

-- | Get the information of the function handle.
mxFuncGetInfo :: FunctionHandle                     -- ^ The target function handle.
              -> IO (Int,
                     String, String,
                     MXUInt,
                     [String], [String], [String],
                     String)                        -- ^ The name of returned function,
                                                    -- it's description, the number of it's
                                                    -- arguments, argument name, type and
                                                    -- descriptions, as well as the return
                                                    -- type of this function.
mxFuncGetInfo handle = do
    (res, name, desc, argc, argv, argtype, argdesc, rettype) <- mxFuncGetInfoImpl handle
    argv' <- peekStringArray argc argv
    argtype' <- peekStringArray argc argtype
    argdesc' <- peekStringArray argc argdesc
    return (res, name, desc, argc, argv', argtype', argdesc', rettype)

-- | Get the argument requirements of the function.
{#fun MXFuncDescribe as mxFuncDescribe
    { id `FunctionHandle'
    , alloca- `MXUInt' peek*
    , alloca- `MXUInt' peek*
    , alloca- `MXUInt' peek*
    , alloca- `Int' peekIntegral*
    } -> `Int' -- ^ The number of NDArrays, scalar variables and mutable NDArrays to be
               -- passed in, and the type mask of this function.
    #}

-- | Invoke a function, the array size of passed in arguments must match the values in the
-- @fun@ function.
{#fun MXFuncInvoke as mxFuncInvoke
    { id `FunctionHandle'           -- ^ The function to invoke.
    , withArray* `[NDArrayHandle]'  -- ^ The normal NDArrays arguments.
    , withArray* `[MXFloat]'        -- ^ The scalar arguments.
    , withArray* `[NDArrayHandle]'  -- ^ The mutable NDArrays arguments.
    } -> `Int' #}

-- | Invoke a function, the array size of passed in arguments must match the values in the
-- @fun@ function.
{#fun MXFuncInvokeEx as mxFuncInvokeEx
    { id `FunctionHandle'           -- ^ The function to invoke.
    , withArray* `[NDArrayHandle]'  -- ^ The normal NDArrays arguments.
    , withArray* `[MXFloat]'        -- ^ The scalar arguments.
    , withArray* `[NDArrayHandle]'  -- ^ The mutable NDArrays arguments.
    , `Int'                         -- ^ Number of keyword parameters.
    , withStringArray* `[String]'   -- ^ Keys for keyword parameters.
    , withStringArray* `[String]'   -- ^ Values for keyword parameters.
    } -> `Int' #}

{#fun MXSymbolListAtomicSymbolCreators as mxSymbolListAtomicSymbolCreatorsImpl
    { alloca- `MXUInt' peek*
    , alloca- `Ptr AtomicSymbolCreator' peek*
    } -> `Int' #}

-- | List all the available @AtomicSymbolCreator@.
mxSymbolListAtomicSymbolCreators
    :: IO (Int, MXUInt, [AtomicSymbolCreator])  -- ^ The number of atomic symbol creators and
                                                -- the atomic symbol creators list.
mxSymbolListAtomicSymbolCreators = do
    (res, n, p) <- mxSymbolListAtomicSymbolCreatorsImpl
    ss <- peekArray (fromIntegral n) p
    return (res, n, ss)

-- | Get the name of an atomic symbol.
{#fun MXSymbolGetAtomicSymbolName as mxSymbolGetAtomicSymbolName
    { id `AtomicSymbolCreator'
    , alloca- `String' peekString*
    } -> `Int' -- ^ Name of the target atomic symbol.
    #}

{#fun MXSymbolGetAtomicSymbolInfo as mxSymbolGetAtomicSymbolInfoImpl
    { id `AtomicSymbolCreator'
    , alloca- `String' peekString*
    , alloca- `String' peekString*
    , alloca- `MXUInt' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    , withStringArray* `[String]'
    , alloca- `String' peekString*
    } -> `Int' #}

-- | Get the detailed information about atomic symbol.
mxSymbolGetAtomicSymbolInfo
    :: AtomicSymbolCreator
    -> [String]                             -- ^ TODO document for this argument.
                                            -- The keyword arguments for specifying variable
                                            -- number of arguments.
    -> IO (Int, String, String, MXUInt,
           [String], [String], [String],
           String)                          -- ^ Return the name and description of the symbol,
                                            -- the name, type and description of it's arguments,
                                            -- as well as the return type of this symbol.
mxSymbolGetAtomicSymbolInfo creator kargs = do
    (res, name, desc, argc, argv, argtype, argdesc, rettype) <- mxSymbolGetAtomicSymbolInfoImpl creator kargs
    argv' <- peekStringArray argc argv
    argtype' <- peekStringArray argc argtype
    argdesc' <- peekStringArray argc argdesc
    return (res, name, desc, argc, argv', argtype', argdesc', rettype)

-- | Create an AtomicSymbol.
{#fun MXSymbolCreateAtomicSymbol as mxSymbolCreateAtomicSymbol
    { id `AtomicSymbolCreator'      -- ^ The atomic symbol creator.
    , id `MXUInt'                   -- ^ The number of parameters.
    , withStringArray* `[String]'   -- ^ The keys of the parameters.
    , withStringArray* `[String]'   -- ^ The values of the parameters.
    , alloca- `SymbolHandle' peek*
    } -> `Int' -- ^ The created symbol.
    #}

-- | Create a Variable Symbol.
{#fun MXSymbolCreateVariable as mxSymbolCreateVariable
    { `String'                      -- ^ Name of the variable.
    , alloca- `SymbolHandle' peek*
    } -> `Int' -- ^ The created variable symbol.
    #}

-- | Create a Symbol by grouping list of symbols together.
{#fun MXSymbolCreateGroup as mxSymbolCreateGroup
    { id `MXUInt'                   -- ^ Number of symbols to be grouped.
    , withArray* `[SymbolHandle]'
    , alloca- `SymbolHandle' peek*  -- ^ Symbols to be added into the new group.
    } -> `Int' -- ^ The created symbol group.
    #}

-- | Load a symbol from a json file.
{#fun MXSymbolCreateFromFile as mxSymbolCreateFromFile
    { `String'                      -- ^ The file name
    , alloca- `SymbolHandle' peek*
    } -> `Int' #}

-- | Load a symbol from a json string.
{#fun MXSymbolCreateFromJSON as mxSymbolCreateFromJSON
    { `String'                      -- ^ The json string.
    , alloca- `SymbolHandle' peek*
    } -> `Int' #}

-- | Save a symbol into a json file.
{#fun MXSymbolSaveToFile as mxSymbolSaveToFile
    { id `SymbolHandle' -- ^ The symbol to save.
    , `String'          -- ^ The target file name.
    } -> `Int' #}

-- | Save a symbol into a json string.
{#fun MXSymbolSaveToJSON as mxSymbolSaveToJSON
    { id `SymbolHandle'             -- ^ The symbol to save.
    , alloca- `String' peekString*
    } -> `Int' -- ^ The result json string.
    #}

-- | Free the symbol handle.
{#fun MXSymbolFree as mxSymbolFree
    { id `SymbolHandle'
    } -> `Int' #}

-- | Copy the symbol to another handle.
{#fun MXSymbolCopy as mxSymbolCopy
    { id `SymbolHandle'
    , alloca- `SymbolHandle' peek*
    } -> `Int' #}

-- | Print the content of symbol, used for debug.
{#fun MXSymbolPrint as mxSymbolPrint
    { id `SymbolHandle'             -- ^ The symbol handle to print.
    , alloca- `String' peekString*  -- ^ The output string.
    } -> `Int' #}

-- | Get string name from symbol
{#fun MXSymbolGetName as mxSymbolGetName
    { id `SymbolHandle'
    , alloca- `String' peekString*
    , alloca- `Int' peekIntegral*
    } -> `Int' -- ^ The name of the symbol and whether the call is successful.
    #}

-- | Get string attribute from symbol.
{#fun MXSymbolGetAttr as mxSymbolGetAttr
    { id `SymbolHandle'             -- ^ The source symbol.
    , `String'                      -- ^ The key of this attribute.
    , alloca- `String' peekString*
    , alloca- `Int' peekIntegral*
    } -> `Int' -- ^ The value of this attribute and whether the call is successful.
    #}

-- | Set string attribute from symbol. Setting attribute to a symbol can affect the semantics
-- (mutable/immutable) of symbolic graph.
{#fun MXSymbolSetAttr as mxSymbolSetAttr
    { id `SymbolHandle' -- ^ The source symbol.
    , `String'          -- ^ The name of the attribute.
    , `String'          -- ^ The value of the attribute.
    } -> `Int' #}

{#fun MXSymbolListAttr as mxSymbolListAttrImpl
    { id `SymbolHandle'
    , alloca- `MXUInt' peekIntegral*
    , alloca- `Ptr (Ptr CChar)' peek*
    } -> `Int' #}

-- | Get all attributes from symbol, including all descendents.
mxSymbolListAttr :: SymbolHandle
                 -> IO (Int, MXUInt, [String])  -- ^ The number of attributes and
                                                -- attributes list.
mxSymbolListAttr symbol = do
    (res, n, p) <- mxSymbolListAttrImpl symbol
    ss <- peekStringArray n p
    return (res, n, ss)

{#fun MXSymbolListAttrShallow as mxSymbolListAttrShallowImpl
    { id `SymbolHandle'
    , alloca- `MXUInt' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    } -> `Int' #}

-- | Get all attributes from symbol, excluding descendents.
mxSymbolListAttrShallow :: SymbolHandle
                        -> IO (Int, MXUInt, [String])   -- ^ The number of attributes and
                                                        -- attributes list.
mxSymbolListAttrShallow symbol = do
    (res, n, p) <- mxSymbolListAttrShallowImpl symbol
    ss <- peekStringArray n p
    return (res, n, ss)

{#fun MXSymbolListArguments as mxSymbolListArgumentsImpl
    { id `SymbolHandle'
    , alloca- `MXUInt' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    } -> `Int' #}

-- | List arguments in the symbol.
mxSymbolListArguments :: SymbolHandle
                      -> IO (Int, MXUInt, [String]) -- ^ The number of arguments and list of
                                                    -- arguments' names.
mxSymbolListArguments symbol = do
    (res, n, p) <- mxSymbolListArgumentsImpl symbol
    ss <- peekStringArray n p
    return (res, n, ss)

{#fun MXSymbolListOutputs as mxSymbolListOutputsImpl
    { id `SymbolHandle'
    , alloca- `MXUInt' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    } -> `Int' #}

-- | List returns in the symbol.
mxSymbolListOutputs :: SymbolHandle
                    -> IO (Int, MXUInt, [String])   -- ^ The number of outputs and list of
                                                    -- outputs' names.
mxSymbolListOutputs symbol = do
    (res, n, p) <- mxSymbolListOutputsImpl symbol
    ss <- peekStringArray n p
    return (res, n, ss)

-- | Get a symbol that contains all the internals.
{#fun MXSymbolGetInternals as mxSymbolGetInternals
    { id `SymbolHandle'
    , alloca- `SymbolHandle' peek*
    } -> `Int' -- ^ The output symbol whose outputs are all the internals.
    #}

-- | Get index-th outputs of the symbol.
{#fun MXSymbolGetOutput as mxSymbolGetOutput
    { id `SymbolHandle'             -- ^ The symbol.
    , id `MXUInt'                   -- ^ Index of the output.
    , alloca- `SymbolHandle' peek*
    } -> `Int' -- ^ The output symbol whose outputs are the index-th symbol.
    #}

{#fun MXSymbolListAuxiliaryStates as mxSymbolListAuxiliaryStatesImpl
    { id `SymbolHandle'
    , alloca- `MXUInt' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    } -> `Int' #}

-- | List auxiliary states in the symbol.
mxSymbolListAuxiliaryStates
    :: SymbolHandle
    -> IO (Int, MXUInt, [String])   -- ^ The output size and the output string array.
mxSymbolListAuxiliaryStates symbol = do
    (res, n, p) <- mxSymbolListAuxiliaryStatesImpl symbol
    ss <- peekStringArray n p
    return (res, n, ss)
