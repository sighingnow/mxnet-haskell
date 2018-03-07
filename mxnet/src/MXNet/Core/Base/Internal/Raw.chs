-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Base.Internal.Raw
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Direct C FFI bindings for <mxnet/c_api.h>.
--
#if __GLASGOW_HASKELL__ >= 801
{-# LANGUAGE Strict #-}
#endif
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}

module MXNet.Core.Base.Internal.Raw where

import Foreign.C.Types
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Ptr
import Foreign.Storable

import C2HS.C.Extra.Marshal

import Data.Typeable
import Control.Exception.Base

{#import MXNet.Core.Types.Internal.Raw #}
import Data.Tuple.Ops

#include <mxnet/c_api.h>

-- | Handle size_t type.
{#typedef size_t CSize#}

-- | Get the string message of last error.
{#fun MXGetLastError as mxGetLastError
    {
    } -> `String' #}

data MXNetError = MXNetError String
    deriving (Typeable, Show)
instance Exception MXNetError

checked :: Unconsable t Int r => IO t -> IO r
checked call = do
    (res, ret) <- uncons <$> call
    if res < 0 
      then do err <- mxGetLastError
              throwIO $ MXNetError err
      else return ret

-------------------------------------------------------------------------------

-- | Seed the global random number generators in mxnet.
{#fun MXRandomSeed as mxRandomSeed
    { `Int'
    } -> `Int' #}

-- | Notify the engine about a shutdown.
{#fun MXNotifyShutdown as mxNotifyShutdown
    {
    } -> `Int' #}

-- | Set up configuration of profiler.
{#fun MXSetProfilerConfig as mxSetProfilerConfig
    { `Int'         -- ^ Mode, indicate the working mode of profiler, record anly symbolic
                    -- operator when mode == 0, record all operator when mode == 1.
    , `String'      -- ^ Filename, where to save trace file.
    } -> `Int' #}

-- | Set up state of profiler.
{#fun MXSetProfilerState as mxSetProfilerState
    { `Int'         -- ^ State, indicate the working state of profiler, profiler not running
                    -- when state == 0, profiler running when state == 1.
    } -> `Int' #}

-- | Save profile and stop profiler.
{#fun MXDumpProfile as mxDumpProfile
    {
    } -> `Int' #}

-------------------------------------------------------------------------------

-- | Create a NDArray handle that is not initialized.
{#fun MXNDArrayCreateNone as mxNDArrayCreateNone
    { alloca- `NDArrayHandle' peekNDArrayHandle*
    } -> `Int' -- ^ The returned NDArrayHandle.
    #}

-- | Create a NDArray with specified shape.
{#fun MXNDArrayCreate as mxNDArrayCreate
    { withArray* `[MXUInt]'         -- ^ The shape of NDArray.
    , id `MXUInt'                   -- ^ The dimension of the shape.
    , `Int'                         -- ^ Device type, specify device we want to take.
    , `Int'                         -- ^ The device id of the specific device.
    , `Int'                         -- ^ Whether to delay allocation until.
    , alloca- `NDArrayHandle' peekNDArrayHandle*
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
    , alloca- `NDArrayHandle' peekNDArrayHandle*
    } -> `Int' -- ^ The returing handle.
    #}

-- | Create a NDArray handle that is loaded from raw bytes.
{#fun MXNDArrayLoadFromRawBytes as mxNDArrayLoadFromRawBytes
    { id `Ptr ()'                   -- ^ The head of the raw bytes.
    , `CSize'                       -- ^ Size of the raw bytes.
    , alloca- `NDArrayHandle' peekNDArrayHandle*
    } -> `Int' #}

-- | Save the NDArray into raw bytes.
{#fun MXNDArraySaveRawBytes as mxNDArraySaveRawBytes
    { withNDArrayHandle* `NDArrayHandle'        -- ^ The NDArray handle.
    , alloca- `CSize' peek*     -- ^ Size of the raw bytes.
    , alloca- `Ptr CChar' peek* -- ^ The head of returning memory bytes.
    } -> `Int' #}

-- | Save list of narray into the file.
{#fun MXNDArraySave as mxNDArraySave
    { `String'                      -- ^ Name of the file.
    , id `MXUInt'                   -- ^ Number of arguments to save.
    , withNDArrayHandleArray* `[NDArrayHandle]'  -- ^ the array of NDArrayHandles to be saved.
    , withStringArray* `[String]'   -- ^ names of the NDArrays to save.
    } -> `Int' #}

{#fun MXNDArrayLoad as mxNDArrayLoadImpl
    { `String'                      -- ^ Name of the file.
    , alloca- `MXUInt' peek*
    , alloca- `Ptr NDArrayHandlePtr' peek*
    , alloca- `MXUInt' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    } -> `Int' #}

-- | Load list of narray from the file.
mxNDArrayLoad :: String                         -- ^ Name of the file.
              -> IO (MXUInt, [NDArrayHandle],
                     MXUInt, [String])          -- ^ The size of ndarray handles, ndarray
                                                -- handles the number of names and the
                                                -- returned names.
mxNDArrayLoad fname = do
    (c1, p1, c2, p2) <- checked $ mxNDArrayLoadImpl fname
    handles <- peekArray (fromIntegral c1) p1 >>= mapM newNDArrayHandle
    names <- peekStringArray c2 p2
    return (c1, handles, c2, names)

-- | Perform a synchronize copy from a continugous CPU memory region.
-- This is useful to copy data from existing memory region that are
-- not wrapped by NDArray (thus dependency not being tracked).
{#fun MXNDArraySyncCopyFromCPU as mxNDArraySyncCopyFromCPU
    { withNDArrayHandle* `NDArrayHandle'    -- ^ The NDArrayHandle.
    , id `Ptr ()'           -- ^ The raw data source to copy from.
    , `CSize'               -- ^ The memory size want to copy from.
    } -> `Int' #}

-- | Perform a synchronize copy to a continugous CPU memory region.
{#fun MXNDArraySyncCopyToCPU as mxNDArraySyncCopyToCPU
    { withNDArrayHandle* `NDArrayHandle'    -- ^ The NDArrayHandle.
    , id `Ptr ()'           -- ^ The raw data source to copy into.
    , `CSize'               -- ^ The memory size want to copy into.
    } -> `Int' #}

-- | Wait until all the pending writes with respect NDArray are finished.
{#fun MXNDArrayWaitToRead as mxNDArrayWaitToRead
    { withNDArrayHandle* `NDArrayHandle'
    } -> `Int' #}

-- | Wait until all the pending read/write with respect NDArray are finished.
{#fun MXNDArrayWaitToWrite as mxNDArrayWaitToWrite
    { withNDArrayHandle* `NDArrayHandle'
    } -> `Int' #}

-- | Wait until all delayed operations in the system is completed.
{#fun MXNDArrayWaitAll as mxNDArrayWaitAll
    {
    } -> `Int' #}

-- | Free the narray handle.
-- {#fun MXNDArrayFree as mxNDArrayFree
--     { id `NDArrayHandle'
--     } -> `Int' #}

-- | Slice the NDArray along axis 0.
{#fun MXNDArraySlice as mxNDArraySlice
    { withNDArrayHandle* `NDArrayHandle'            -- ^ The handle to the NDArray.
    , id `MXUInt'                   -- ^ The beginning index of slice.
    , id `MXUInt'                   -- ^ The ending index of slice.
    , alloca- `NDArrayHandle' peekNDArrayHandle*
    } -> `Int' -- ^ The NDArrayHandle of sliced NDArray.
    #}

-- | Index the NDArray along axis 0.
{#fun MXNDArrayAt as mxNDArrayAt
    { withNDArrayHandle* `NDArrayHandle'            -- ^ The handle to the NDArray.
    , id `MXUInt'                   -- ^ The index.
    , alloca- `NDArrayHandle' peekNDArrayHandle*
    } -> `Int' -- ^ The NDArrayHandle of output NDArray.
    #}

-- | Reshape the NDArray.
{#fun MXNDArrayReshape as mxNDArrayReshape
    { withNDArrayHandle* `NDArrayHandle'            -- ^ The handle to the NDArray.
    , `Int'                         -- ^ Number of dimensions of new shape.
    , withIntegralArray* `[Int]'    -- ^ New sizes of every dimension.
    , alloca- `NDArrayHandle' peekNDArrayHandle*
    } -> `Int' -- ^ The new shape data and the NDArrayHandle of reshaped NDArray.
    #}

{#fun MXNDArrayGetShape as mxNDArrayGetShapeImpl
    { withNDArrayHandle* `NDArrayHandle'
    , alloca- `MXUInt' peek*
    , alloca- `Ptr MXUInt' peek*
    } -> `Int' #}

-- Get the shape of the array.
mxNDArrayGetShape :: NDArrayHandle
                  -> IO (MXUInt, [MXUInt]) -- ^ The output dimension and it's shape.
mxNDArrayGetShape handle = do
    (d, p) <- checked $ mxNDArrayGetShapeImpl handle
    shapes <- peekArray (fromIntegral d) p
    return (d, shapes)

-- | Get the content of the data in NDArray.
{#fun MXNDArrayGetData as mxNDArrayGetData
    { withNDArrayHandle* `NDArrayHandle'            -- ^ The NDArray handle.
    , alloca- `Ptr ()' peek*
    } -> `Int' -- ^ Pointer holder to get pointer of data.
    #}

-- | Get the type of the data in NDArray
{#fun MXNDArrayGetDType as mxNDArrayGetDType
    { withNDArrayHandle* `NDArrayHandle'            -- ^ The NDArray handle.
    , alloca- `Int' peekIntegral*
    } -> `Int' -- ^ The type of data in this NDArray handle.
    #}

-- | Get the context of the NDArray.
{#fun MXNDArrayGetContext as mxNDArrayGetContext
    { withNDArrayHandle* `NDArrayHandle'            -- ^ The NDArray handle.
    , alloca- `Int' peekIntegral*
    , alloca- `Int' peekIntegral*
    } -> `Int' -- ^ The device type and device id.
    #}

-------------------------------------------------------------------------------

{#fun MXListFunctions as mxListFunctionsImpl
    { alloca- `MXUInt' peek*
    , alloca- `Ptr FunctionHandle' peek*
    } -> `Int' #}

-- | List all the available functions handles.
mxListFunctions :: IO ([FunctionHandle]) -- ^ The output function handle array.
mxListFunctions = do
    (c, p) <- checked mxListFunctionsImpl
    peekArray (fromIntegral c) p

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
              -> IO (String, String,
                     MXUInt,
                     [String], [String], [String],
                     String)                        -- ^ The name of returned function,
                                                    -- it's description, the number of it's
                                                    -- arguments, argument name, type and
                                                    -- descriptions, as well as the return
                                                    -- type of this function.
mxFuncGetInfo handle = do
    (name, desc, argc, argv, argtype, argdesc, rettype) <- checked $ mxFuncGetInfoImpl handle
    argv' <- peekStringArray argc argv
    argtype' <- peekStringArray argc argtype
    argdesc' <- peekStringArray argc argdesc
    return (name, desc, argc, argv', argtype', argdesc', rettype)

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
    , withNDArrayHandleArray* `[NDArrayHandle]'  -- ^ The normal NDArrays arguments.
    , withArray* `[MXFloat]'        -- ^ The scalar arguments.
    , withNDArrayHandleArray* `[NDArrayHandle]'  -- ^ The mutable NDArrays arguments.
    } -> `Int' #}

-- | Invoke a function, the array size of passed in arguments must match the values in the
-- @fun@ function.
{#fun MXFuncInvokeEx as mxFuncInvokeEx
    { id `FunctionHandle'           -- ^ The function to invoke.
    , withNDArrayHandleArray* `[NDArrayHandle]'  -- ^ The normal NDArrays arguments.
    , withArray* `[MXFloat]'        -- ^ The scalar arguments.
    , withNDArrayHandleArray* `[NDArrayHandle]'  -- ^ The mutable NDArrays arguments.
    , `Int'                         -- ^ Number of keyword parameters.
    , withStringArray* `[String]'   -- ^ Keys for keyword parameters.
    , withStringArray* `[String]'   -- ^ Values for keyword parameters.
    } -> `Int' #}

{#fun MXImperativeInvoke as mxImperativeInvokeImpl
    { id `AtomicSymbolCreator'      -- ^ Creator of the OP.
    , `Int'
    , withNDArrayHandleArray* `[NDArrayHandle]'
    , id `Ptr CInt'
    , id `Ptr (Ptr NDArrayHandlePtr)'
    , `Int'
    , withStringArray* `[String]'
    , withStringArray* `[String]'
    } -> `Int' #}

-- | Invoke a nnvm op and imperative function.
mxImperativeInvoke :: AtomicSymbolCreator       -- ^ Creator/Handler of the OP.
                   -> [NDArrayHandle]           -- ^ Input NDArrays.
                   -> [(String, String)]        -- ^ Keywords parameters.
                   -> Maybe [NDArrayHandle]     -- ^ Original given output handles array.
                   -> IO [NDArrayHandle]        -- ^ Return NDArrays as result.
mxImperativeInvoke creator inputs params outputs = do
    let (keys, values) = unzip params
        ninput = length inputs
        nparam = length params
    case outputs of
        Nothing -> alloca $ \pn ->
            alloca $ \pp -> do
                poke pn 0
                poke pp nullPtr
                checked $ mxImperativeInvokeImpl creator ninput inputs pn pp nparam keys values
                n' <- peek pn
                p' <- peek pp
                if n' == 0 
                    then return []
                    else do
                        pa <- peekArray (fromIntegral n') p' 
                        mapM newNDArrayHandle pa
        Just out -> alloca $ \pn ->
            alloca $ \pp -> do
                n' <- withNDArrayHandleArray out $ \p' -> do
                    poke pn (fromIntegral $ length out)
                    poke pp p'
                    checked $ mxImperativeInvokeImpl creator ninput inputs pn pp nparam keys values
                    peek pn
                return $ take (fromIntegral n') out

-------------------------------------------------------------------------------

{#fun MXListAllOpNames as mxListAllOpNamesImpl
    { alloca- `MXUInt' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    } -> `Int' #}

-- | List all the available operator names, include entries.
mxListAllOpNames :: IO [String]
mxListAllOpNames = do
    (n, p) <- checked mxListAllOpNamesImpl
    peekStringArray (fromIntegral n :: Int) p

{#fun MXSymbolListAtomicSymbolCreators as mxSymbolListAtomicSymbolCreatorsImpl
    { alloca- `MXUInt' peek*
    , alloca- `Ptr AtomicSymbolCreator' peek*
    } -> `Int' #}

-- | List all the available @AtomicSymbolCreator@.
mxSymbolListAtomicSymbolCreators
    :: IO [AtomicSymbolCreator]  -- ^ The atomic symbol creators list.
mxSymbolListAtomicSymbolCreators = do
    (n, p) <- checked mxSymbolListAtomicSymbolCreatorsImpl
    peekArray (fromIntegral n) p

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
    , alloca- `String' peekString*
    , alloca- `String' peekString*
    } -> `Int' #}

-- | Get the detailed information about atomic symbol.
mxSymbolGetAtomicSymbolInfo
    :: AtomicSymbolCreator
    -> IO (String, String, MXUInt,
           [String], [String], [String],
           String, String)                  -- ^ Return the name and description of the symbol,
                                            -- the name, type and description of it's arguments,
                                            -- the keyword argument for specifying variable number
                                            -- of arguments, as well as the return type of this
                                            -- symbol.
mxSymbolGetAtomicSymbolInfo creator = do
    -- Documentation for kargs: https://github.com/dmlc/mxnet/blob/master/include/mxnet/c_api.h#L555
    (name, desc, argc, argv, argtype, argdesc, kargs, rettype) <- checked $ mxSymbolGetAtomicSymbolInfoImpl creator
    argv' <- peekStringArray argc argv
    argtype' <- peekStringArray argc argtype
    argdesc' <- peekStringArray argc argdesc
    return (name, desc, argc, argv', argtype', argdesc', kargs, rettype)

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
                 -> IO [String]  -- ^ The attributes list.
mxSymbolListAttr symbol = do
    (n, p) <- checked $ mxSymbolListAttrImpl symbol
    peekStringArray n p

{#fun MXSymbolListAttrShallow as mxSymbolListAttrShallowImpl
    { id `SymbolHandle'
    , alloca- `MXUInt' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    } -> `Int' #}

-- | Get all attributes from symbol, excluding descendents.
mxSymbolListAttrShallow :: SymbolHandle
                        -> IO [String]   -- ^ The attributes list.
mxSymbolListAttrShallow symbol = do
    (n, p) <- checked $ mxSymbolListAttrShallowImpl symbol
    peekStringArray n p

{#fun MXSymbolListArguments as mxSymbolListArgumentsImpl
    { id `SymbolHandle'
    , alloca- `MXUInt' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    } -> `Int' #}

-- | List arguments in the symbol.
mxSymbolListArguments :: SymbolHandle
                      -> IO [String] -- ^ List of arguments' names.
mxSymbolListArguments symbol = do
    (n, p) <- checked $ mxSymbolListArgumentsImpl symbol
    peekStringArray n p

{#fun MXSymbolListOutputs as mxSymbolListOutputsImpl
    { id `SymbolHandle'
    , alloca- `MXUInt' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    } -> `Int' #}

-- | List returns in the symbol.
mxSymbolListOutputs :: SymbolHandle
                    -> IO [String]   -- ^ The outputs' names.
mxSymbolListOutputs symbol = do
    (n, p) <- checked $ mxSymbolListOutputsImpl symbol
    peekStringArray n p

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
    -> IO [String]   -- ^ The output string array.
mxSymbolListAuxiliaryStates symbol = do
    (n, p) <- checked $ mxSymbolListAuxiliaryStatesImpl symbol
    peekStringArray n p

-- | Compose the symbol on other symbols.
{#fun MXSymbolCompose as mxSymbolCompose
    { id `SymbolHandle'             -- ^ The symbol to apply.
    , `String'                      -- ^ Name of the symbol.
    , id `MXUInt'                   -- ^ Number of arguments.
    , withStringArray* `[String]'   -- ^ Key of keyword arguments, optional.
    , withArray* `[SymbolHandle]'   -- ^ Arguments.
    } -> `Int' #}

-- | Get the gradient graph of the symbol.
{#fun MXSymbolGrad as mxSymbolGrad
    { id `SymbolHandle'             -- ^ The symbol to get gradient.
    , id `MXUInt'                   -- ^ Number of arguments to get gradient.
    , withStringArray* `[String]'   -- ^ Names of the arguments to get gradient.
    , alloca- `SymbolHandle' peek*
    } -> `Int' -- ^ Return the symbol that has gradient.
    #}

{#fun MXSymbolInferShape as mxSymbolInferShapeImpl
    { id `SymbolHandle'
    , id `MXUInt'
    , withStringArray* `[String]'
    , withIntegralArray* `[Int]'
    , withIntegralArray* `[Int]'
    , alloca- `MXUInt' peek*
    , alloca- `Ptr MXUInt' peek*
    , alloca- `Ptr (Ptr MXUInt)' peek*
    , alloca- `MXUInt' peek*
    , alloca- `Ptr MXUInt' peek*
    , alloca- `Ptr (Ptr MXUInt)' peek*
    , alloca- `MXUInt' peek*
    , alloca- `Ptr MXUInt' peek*
    , alloca- `Ptr (Ptr MXUInt)' peek*
    , alloca- `Int' peekIntegral*
    } -> `Int' #}

-- | Infer shape of unknown input shapes given the known one.
mxSymbolInferShape
    :: SymbolHandle                          -- ^ Symbol handle.
    -> [String]                              -- ^ Keys of keyword arguments, optional.
    -> [Int]                                 -- ^ The head pointer of the rows in CSR.
    -> [Int]                                 -- ^ The content of the CSR.
    -> IO ([[Int]], [[Int]], [[Int]])        -- ^ Return the in, out and auxiliary
                                             -- shape size, ndim and data (array
                                             -- of pointers to head of the input
                                             -- shape), and whether infer shape
                                             -- completes or more information is
                                             -- needed.
mxSymbolInferShape sym keys ind shapedata = do
    let argc = fromIntegral (length keys)   -- Number of input arguments.
    -- Notice: the complete result are ignored for simplification.
    (in_size, in_ndim, in_data, out_size, out_ndim, out_data, aux_size, aux_ndim, aux_data, _) <- checked $ mxSymbolInferShapeImpl sym argc keys ind shapedata
    in_ndim' <- peekIntegralArray (fromIntegral in_size) in_ndim
    in_data' <- peekArray (fromIntegral in_size) in_data
    in_data'' <- mapM (uncurry peekIntegralArray) (zip in_ndim' in_data')
    out_ndim' <- peekIntegralArray (fromIntegral out_size) out_ndim
    out_data' <- peekArray (fromIntegral out_size) out_data
    out_data'' <- mapM (uncurry peekIntegralArray) (zip out_ndim' out_data')
    aux_ndim' <- peekIntegralArray (fromIntegral aux_size) aux_ndim
    aux_data' <- peekArray (fromIntegral aux_size) aux_data
    aux_data'' <- mapM (uncurry peekIntegralArray) (zip aux_ndim' aux_data')
    return (in_data'', out_data'', aux_data'')

{#fun MXSymbolInferShapePartial as mxSymbolInferShapePartialImpl
    { id `SymbolHandle'
    , id `MXUInt'
    , withStringArray* `[String]'
    , withIntegralArray* `[Int]'
    , withIntegralArray* `[Int]'
    , alloca- `MXUInt' peek*
    , alloca- `Ptr MXUInt' peek*
    , alloca- `Ptr (Ptr MXUInt)' peek*
    , alloca- `MXUInt' peek*
    , alloca- `Ptr MXUInt' peek*
    , alloca- `Ptr (Ptr MXUInt)' peek*
    , alloca- `MXUInt' peek*
    , alloca- `Ptr MXUInt' peek*
    , alloca- `Ptr (Ptr MXUInt)' peek*
    , alloca- `Int' peekIntegral*
    } -> `Int' #}

-- | Partially infer shape of unknown input shapes given the known one.
mxSymbolInferShapePartial
    :: SymbolHandle                         -- ^ Symbol handle.
    -> [String]                              -- ^ Keys of keyword arguments, optional.
    -> [Int]                             -- ^ The head pointer of the rows in CSR.
    -> [Int]                             -- ^ The content of the CSR.
    -> IO ([[Int]], [[Int]], [[Int]])    -- ^ Return the in, out and auxiliary array's
                                            -- shape size, ndim and data (array of pointers
                                            -- to head of the input shape), and whether
                                            -- infer shape completes or more information is
                                            -- needed.
mxSymbolInferShapePartial sym keys ind shapedata = do
    let argc = fromIntegral (length keys)   -- Number of input arguments.
    -- Notice: the complete result are ignored for simplification.
    (in_size, in_ndim, in_data, out_size, out_ndim, out_data, aux_size, aux_ndim, aux_data, _) <- checked $ mxSymbolInferShapePartialImpl sym argc keys ind shapedata
    in_ndim' <- peekIntegralArray (fromIntegral in_size) in_ndim
    in_data' <- peekArray (fromIntegral in_size) in_data
    in_data'' <- mapM (uncurry peekIntegralArray) (zip in_ndim' in_data')
    out_ndim' <- peekIntegralArray (fromIntegral out_size) out_ndim
    out_data' <- peekArray (fromIntegral out_size) out_data
    out_data'' <- mapM (uncurry peekIntegralArray) (zip out_ndim' out_data')
    aux_ndim' <- peekIntegralArray (fromIntegral aux_size) aux_ndim
    aux_data' <- peekArray (fromIntegral aux_size) aux_data
    aux_data'' <- mapM (uncurry peekIntegralArray) (zip aux_ndim' aux_data')
    return (in_data'', out_data'', aux_data'')

{#fun MXSymbolInferType as mxSymbolInferTypeImpl
    { id `SymbolHandle'             -- ^ Symbol handle.
    , id `MXUInt'                   -- ^ Number of input arguments.
    , withStringArray* `[String]'   -- ^ Key of keyword arguments, optional.
    , withIntegralArray* `[Int]'    -- ^ The content of the CSR.
    , alloca- `MXUInt' peek*
    , alloca- `Ptr CInt' peek*
    , alloca- `MXUInt' peek*
    , alloca- `Ptr CInt' peek*
    , alloca- `MXUInt' peek*
    , alloca- `Ptr CInt' peek*
    , alloca- `Int' peekIntegral*
    } -> `Int' -- ^ Return the size and an array of pointers to head the input, output and
               -- auxiliary type, as well as whether infer type completes or more information
               -- is needed.
    #}

-- | Infer type of unknown input types given the known one.
mxSymbolInferType :: SymbolHandle                   -- ^ Symbol handle.
                  -> [String]                       -- ^ Input arguments.
                  -> IO ([Int], [Int], [Int])       -- ^ Return arg_types, out_types and aux_types.
mxSymbolInferType handle args = do
    let nargs = fromIntegral (length args)
        csr = []
    -- Notice: the complete result are ignored for simplification.
    (narg, parg, nout, pout, naux, paux, _) <- checked $ mxSymbolInferTypeImpl handle nargs args csr
    args <- peekIntegralArray (fromIntegral narg) parg
    outs <- peekIntegralArray (fromIntegral nout) pout
    auxs <- peekIntegralArray (fromIntegral naux) paux
    return (args, outs, auxs)

-------------------------------------------------------------------------------

-- | Delete the executor.
-- {#fun MXExecutorFree as mxExecutorFree
--     { id `ExecutorHandle' -- ^ The executor handle.
--     } -> `Int' #}

-- | Print the content of execution plan, used for debug.
{#fun MXExecutorPrint as mxExecutorPrint
    { withExecutorHandle* `ExecutorHandle'          -- ^ The executor handle.
    , alloca- `String' peekString*  -- ^ Pointer to hold the output string of the printing.
    } -> `Int' #}

-- | Executor forward method.
{#fun MXExecutorForward as mxExecutorForward
    { withExecutorHandle* `ExecutorHandle'   -- ^ The executor handle.
    , `Int'                 -- ^ int value to indicate whether the forward pass is for
                            -- evaluation.
    } -> `Int' #}

-- | Excecutor run backward.
{#fun MXExecutorBackward as mxExecutorBackward
    { withExecutorHandle* `ExecutorHandle'          -- ^ The executor handle.
    , id `MXUInt'                                   -- ^ Length.
    , withNDArrayHandleArray* `[NDArrayHandle]'     -- ^ NDArray handle for heads' gradient.
    } -> `Int' #}

{#fun MXExecutorOutputs as mxExecutorOutputsImpl
    { withExecutorHandle* `ExecutorHandle'           -- ^ The executor handle.
    , alloca- `MXUInt' peek*                         -- ^ NDArray vector size.
    , alloca- `Ptr NDArrayHandlePtr' peek*
    } -> `Int' #}

-- | Get executor's head NDArray.
mxExecutorOutputs :: ExecutorHandle             -- ^ The executor handle.
                  -> IO ([NDArrayHandle])       -- ^ The handles for outputs.
mxExecutorOutputs handle = do
    (c, p) <- checked $ mxExecutorOutputsImpl handle
    handles <- peekArray (fromIntegral c) p >>= mapM newNDArrayHandle
    return handles

-- | Generate Executor from symbol.
{#fun MXExecutorBind as mxExecutorBind
    { id `SymbolHandle'                 -- ^ The symbol handle.
    , `Int'                             -- ^ Device type.
    , `Int'                             -- ^ Device id.
    , id `MXUInt'                       -- ^ Length of arrays in arguments.
    , withNDArrayHandleArray* `[NDArrayHandle]'      -- ^ In array.
    , withNDArrayHandleArray* `[NDArrayHandle]'      -- ^ Grads handle array.
    , withArray* `[MXUInt]'             -- ^ Grad req array.
    , id `MXUInt'                       -- ^ Length of auxiliary states.
    , withNDArrayHandleArray* `[NDArrayHandle]'      -- ^ Auxiliary states array.
    , alloca- `ExecutorHandle' peekExecutorHandle*    -- ^ Output executor handle.
    } -> `Int' #}

-- | Generate Executor from symbol. This is advanced function, allow specify group2ctx map.
-- The user can annotate "ctx_group" attribute to name each group.
{#fun MXExecutorBindX as mxExecutorBindX
    { id `SymbolHandle'                 -- ^ The symbol handle.
    , `Int'                             -- ^ Device type of default context.
    , `Int'                             -- ^ Device id of default context.
    , id `MXUInt'                       -- ^ Size of group2ctx map.
    , withStringArray* `[String]'       -- ^ Keys of group2ctx map.
    , withIntegralArray* `[Int]'        -- ^ Device type of group2ctx map.
    , withIntegralArray* `[Int]'        -- ^ Device id of group2ctx map.
    , id `MXUInt'                       -- ^ Length of arrays in arguments.
    , withNDArrayHandleArray* `[NDArrayHandle]'      -- ^ In array.
    , withNDArrayHandleArray* `[NDArrayHandle]'      -- ^ Grads handle array.
    , withArray* `[MXUInt]'             -- ^ Grad req array.
    , id `MXUInt'                       -- ^ Length of auxiliary states.
    , withNDArrayHandleArray* `[NDArrayHandle]'      -- ^ Auxiliary states array.
    , alloca- `ExecutorHandle' peekExecutorHandle*    -- ^ Output executor handle.
    } -> `Int' #}

-- | Generate Executor from symbol. This is advanced function, allow specify group2ctx map.
-- The user can annotate "ctx_group" attribute to name each group.
{#fun MXExecutorBindEX as mxExecutorBindEX
    { id `SymbolHandle'                 -- ^ The symbol handle.
    , `Int'                             -- ^ Device type of default context.
    , `Int'                             -- ^ Device id of default context.
    , id `MXUInt'                       -- ^ Size of group2ctx map.
    , withStringArray* `[String]'       -- ^ Keys of group2ctx map.
    , withIntegralArray* `[Int]'        -- ^ Device type of group2ctx map.
    , withIntegralArray* `[Int]'        -- ^ Device id of group2ctx map.
    , id `MXUInt'                       -- ^ Length of arrays in arguments.
    , withNDArrayHandleArray* `[NDArrayHandle]'      -- ^ In array.
    , withNDArrayHandleArray* `[NDArrayHandle]'      -- ^ Grads handle array.
    , withArray* `[MXUInt]'             -- ^ Grad req array.
    , id `MXUInt'                       -- ^ Length of auxiliary states.
    , withNDArrayHandleArray* `[NDArrayHandle]'      -- ^ Auxiliary states array.
    , withExecutorHandle* `ExecutorHandle'               -- ^ Input executor handle for memory sharing.
    , alloca- `ExecutorHandle' peekExecutorHandle*    -- ^ Output executor handle.
    } -> `Int' #}

-- | Set a call back to notify the completion of operation.
{#fun MXExecutorSetMonitorCallback as mxExecutorSetMonitorCallback
    { withExecutorHandle* `ExecutorHandle'           -- ^ The executor handle.
    , id `ExecutorMonitorCallback'
    , id `Ptr ()'
    } -> `Int' #}

-------------------------------------------------------------------------------

{#fun MXListDataIters as mxListDataItersImpl
    { alloca- `MXUInt' peek*
    , alloca- `Ptr DataIterCreator' peek*
    } -> `Int' #}

-- | List all the available iterator entries.
mxListDataIters :: IO [DataIterCreator] -- ^ The output iterator entries.
mxListDataIters = do
    (c, p) <- checked $ mxListDataItersImpl
    peekArray (fromIntegral c) p

-- | Init an iterator, init with parameters the array size of passed in arguments.
{#fun MXDataIterCreateIter as mxDataIterCreateIter
    { id `DataIterCreator'              -- ^ The handle pointer to the data iterator.
    , id `MXUInt'                       -- ^ Size of arrays in arguments.
    , withStringArray* `[String]'       -- ^ Parameter keys.
    , withStringArray* `[String]'       -- ^ Parameter values.
    , alloca- `DataIterHandle' peek*    -- ^ Resulting iterator.
    } -> `Int' #}

{#fun MXDataIterGetIterInfo as mxDataIterGetIterInfoImpl
    { id `DataIterCreator'              -- ^ The handle pointer to the data iterator.
    , alloca- `String' peekString*
    , alloca- `String' peekString*
    , alloca- `MXUInt' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    } -> `Int' #}

-- | Get the detailed information about data iterator.
mxDataIterGetIterInfo :: DataIterCreator                    -- ^ The handle pointer to the
                                                            -- data iterator.
                      -> IO (String, String,
                             MXUInt,
                             [String], [String], [String])  -- ^ Return the name and description
                                                            -- of the data iter creator,
                                                            -- the name, type and description of
                                                            -- it's arguments, as well as the
                                                            -- return type of this symbol.
mxDataIterGetIterInfo creator = do
    (name, desc, argc, argv, argtype, argdesc) <- checked $ mxDataIterGetIterInfoImpl creator
    argv' <- peekStringArray argc argv
    argtype' <- peekStringArray argc argtype
    argdesc' <- peekStringArray argc argdesc
    return (name, desc, argc, argv', argtype', argdesc')

-- | Get the detailed information about data iterator.

-- | Free the handle to the IO module.
{#fun MXDataIterFree as mxDataIterFree
    { id `DataIterHandle'   -- ^ The handle pointer to the data iterator.
    } -> `Int' #}

-- | Move iterator to next position.
{#fun MXDataIterNext as mxDataIterNext
    { id `DataIterHandle'           -- ^ The handle pointer to the data iterator.
    , alloca- `Int' peekIntegral*   -- ^ Return value of next.
    } -> `Int' #}

-- | Call iterator.Reset.
{#fun MXDataIterBeforeFirst as mxDataIterBeforeFirst
    { id `DataIterHandle'   -- ^ The handle pointer to the data iterator.
    } -> `Int' #}

-- | Get the handle to the NDArray of underlying data.
{#fun MXDataIterGetData as mxDataIterGetData
    { id `DataIterHandle'           -- ^ The handle pointer to the data iterator.
    , alloca- `NDArrayHandle' peekNDArrayHandle* -- ^ Handle to the underlying data NDArray.
    } -> `Int' #}

#ifdef mingw32_HOST_OS
{#fun MXDataIterGetIndex as mxDataIterGetIndexImpl
    { id `DataIterHandle'               -- ^ The handle pointer to the data iterator.
    , alloca- `Ptr CULLong' peek*       -- ^ The output indices.
    , alloca- `CULLong' peekIntegral*   -- ^ Size of output array.
    } -> `Int' #}
#else
{#fun MXDataIterGetIndex as mxDataIterGetIndexImpl
    { id `DataIterHandle'               -- ^ The handle pointer to the data iterator.
    , alloca- `Ptr CULong' peek*        -- ^ The output indices.
    , alloca- `CULong' peekIntegral*    -- ^ Size of output array.
    } -> `Int' #}
#endif

-- | Get the image index by array.
mxDataIterGetIndex :: DataIterHandle        -- ^ The handle pointer to the data iterator.
#ifdef mingw32_HOST_OS
                   -> IO ([CULLong])   -- ^ Output indices of the array.
#else
                   -> IO ([CULong])    -- ^ Output indices of the array.
#endif
mxDataIterGetIndex creator = do
    (p, c) <- checked $ mxDataIterGetIndexImpl creator
    peekArray (fromIntegral c) p

-- | Get the padding number in current data batch.
{#fun MXDataIterGetPadNum as mxDataIterGetPadNum
    { id `DataIterHandle'           -- ^ The handle pointer to the data iterator.
    , alloca- `Int' peekIntegral*   -- ^ Pad number.
    } -> `Int' #}

-- | Get the handle to the NDArray of underlying label.
{#fun MXDataIterGetLabel as mxDataIterGetLabel
    { id `DataIterHandle'           -- ^ The handle pointer to the data iterator.
    , alloca- `NDArrayHandle' peekNDArrayHandle* -- ^ The handle to underlying label NDArray.
    } -> `Int' #}

-------------------------------------------------------------------------------

-- | Initialized ps-lite environment variables.
{#fun MXInitPSEnv as mxInitPSEnv
    { id `MXUInt'                   -- ^ Number of variables to initialize.
    , withStringArray* `[String]'   -- ^ Environment keys.
    , withStringArray* `[String]'   -- ^ Environment values.
    } -> `Int' #}

-- | Create a kvstore.
{#fun MXKVStoreCreate as mxKVStoreCreate
    { `String'                      -- ^ The type of KVStore.
    , alloca- `KVStoreHandle' peek* -- ^ The output KVStore.
    } -> `Int' #}

-- | Delete a KVStore handle.
{#fun MXKVStoreFree as mxKVStoreFree
    { id `KVStoreHandle'    -- ^ Handle to the kvstore.
    } -> `Int' #}

-- | Init a list of (key,value) pairs in kvstore.
{#fun MXKVStoreInit as mxKVStoreInit
    { id `KVStoreHandle'            -- ^ Handle to the kvstore.
    , id `MXUInt'                   -- ^ The number of key-value pairs.
    , withIntegralArray* `[Int]'    -- ^ The list of keys.
    , withNDArrayHandleArray* `[NDArrayHandle]'  -- ^ The list of values.
    } -> `Int' #}

-- | Push a list of (key,value) pairs to kvstore.
{#fun MXKVStorePush as mxKVStorePush
    { id `KVStoreHandle'            -- ^ Handle to the kvstore.
    , id `MXUInt'                   -- ^ The number of key-value pairs.
    , withIntegralArray* `[Int]'    -- ^ The list of keys.
    , withNDArrayHandleArray* `[NDArrayHandle]'  -- ^ The list of values.
    , `Int'                         -- ^ The priority of the action.
    } -> `Int' #}

-- | FIXME Pull a list of (key, value) pairs from the kvstore.
{#fun MXKVStorePull as mxKVStorePull
    { id `KVStoreHandle'            -- ^ Handle to the kvstore.
    , id `MXUInt'                   -- ^ The number of key-value pairs.
    , withIntegralArray* `[Int]'    -- ^ The list of keys.
    , withNDArrayHandleArray* `[NDArrayHandle]'  -- ^ The list of values.
    , `Int'                         -- ^ The priority of the action.
    } -> `Int' #}

-- | FIXME Register an push updater.
mxKVStoreSetUpdater = undefined
{-
{#fun  as
    { id `KVStoreHandle'
    , id `MXUInt'
    } -> `Int' #}
-}

-- | Get the type of the kvstore.
{#fun MXKVStoreGetType as mxKVStoreGetType
    { id `KVStoreHandle'                -- ^ Handle to the KVStore.
    , alloca- `String' peekString*   -- ^ A string type.
    } -> `Int' #}

-------------------------------------------------------------------------------

-- | The rank of this node in its group, which is in [0, GroupSize).
{#fun MXKVStoreGetRank as mxKVStoreGetRank
    { id `KVStoreHandle'            -- ^ Handle to the KVStore.
    , alloca- `Int' peekIntegral*   -- ^ The node rank.
    } -> `Int' #}

-- | The number of nodes in this group, which is
--
--      * number of workers if if `IsWorkerNode() == true`,
--      * number of servers if if `IsServerNode() == true`,
--      * 1 if `IsSchedulerNode() == true`.
{#fun MXKVStoreGetGroupSize as mxKVStoreGetGroupSize
    { id `KVStoreHandle'            -- ^ Handle to the KVStore.
    , alloca- `Int' peekIntegral*   -- ^ The group size.
    } -> `Int' #}

-- | Return whether or not this process is a worker node.
{#fun MXKVStoreIsWorkerNode as mxKVStoreIsWorkerNode
    { alloca- `Int' peekIntegral*   -- ^ Return 1 for yes, 0 for no.
    } -> `Int' #}

-- | Return whether or not this process is a server node.
{#fun MXKVStoreIsServerNode as mxKVStoreIsServerNode
    { alloca- `Int' peekIntegral*   -- ^ Return 1 for yes, 0 for no.
    } -> `Int' #}

-- | Return whether or not this process is a scheduler node.
{#fun MXKVStoreIsSchedulerNode as mxKVStoreIsSchedulerNode
    { alloca- `Int' peekIntegral*   -- ^ Return 1 for yes, 0 for no.
    } -> `Int' #}

-- | Global barrier among all worker machines.
{#fun MXKVStoreBarrier as mxKVStoreBarrier
    { id `KVStoreHandle'    -- ^ Handle to the KVStore.
    } -> `Int' #}

-- | Whether to do barrier when finalize.
{#fun MXKVStoreSetBarrierBeforeExit as mxKVStoreSetBarrierBeforeExit
    { id `KVStoreHandle'    -- ^ Handle to the KVStore.
    , `Int'                 -- ^ Whether to do barrier when kvstore finalize
    } -> `Int' #}

-- | FIXME  Run as server (or scheduler).
mxKVStoreRunServer = undefined
{-
{#fun MXKVStoreRunServer as mxKVStoreRunServer
    { id `KVStoreHandle'
    , id `MXUInt'
    } -> `Int' #}
-}

-- | Send a command to all server nodes.
{#fun MXKVStoreSendCommmandToServers as mxKVStoreSendCommmandToServers
    { id `KVStoreHandle'    -- ^ Handle to the KVStore.
    , `Int'                 -- ^ The head of the command.
    , `String'              -- ^ The body of the command.
    } -> `Int' #}

-- | Get the number of ps dead node(s) specified by {node_id}.
{#fun MXKVStoreGetNumDeadNode as mxKVStoreGetNumDeadNode
    { id `KVStoreHandle'            -- ^ Handle to the kvstore.
    , `Int'                         -- ^ node id, can be a node group or a single node.
                                    -- kScheduler = 1, kServerGroup = 2, kWorkerGroup = 4
    , alloca- `Int' peekIntegral*   -- ^ Ouptut number of dead nodes.
    , `Int'                         -- ^ A node fails to send heartbeart in {timeout_sec}
                                    -- seconds will be presumed as 'dead'
    } -> `Int' #}

-- | Create a RecordIO writer object.
{#fun MXRecordIOWriterCreate as mxRecordIOWriterCreate
    { `String'                          -- ^ Path to file.
    , alloca- `RecordIOHandle' peek*    -- ^ The created object.
    } -> `Int' #}

-- | Delete a RecordIO writer object.
{#fun MXRecordIOWriterFree as mxRecordIOWriterFree
    { id `RecordIOHandle'   -- ^ Handle to RecordIO object.
    } -> `Int' #}

-- | Write a record to a RecordIO object.
{#fun MXRecordIOWriterWriteRecord as mxRecordIOWriterWriteRecord
    { id `RecordIOHandle'   -- ^ Handle to RecordIO object.
    , id `Ptr CChar'        -- ^ Buffer to write.
    , `CSize'               -- ^ Size of buffer.
    } -> `Int' #}

-- | Get the current writer pointer position.
{#fun MXRecordIOWriterTell as mxRecordIOWriterTell
    { id `RecordIOHandle'   -- ^ Handle to RecordIO object.
    , id `Ptr CSize'        -- ^ Handle to output position.
    } -> `Int' #}

-- | Create a RecordIO reader object.
{#fun MXRecordIOReaderCreate as mxRecordIOReaderCreate
    { `String'                          -- ^ Path to file.
    , alloca- `RecordIOHandle' peek*    -- ^ Handle pointer to the created object.
    } -> `Int' #}

-- | Delete a RecordIO reader object.
{#fun MXRecordIOReaderFree as mxRecordIOReaderFree
    { id `RecordIOHandle'   -- ^ Handle to RecordIO object.
    } -> `Int' #}


-- | Write a record to a RecordIO object.
{#fun MXRecordIOReaderReadRecord as mxRecordIOReaderReadRecord
    { id `RecordIOHandle'   -- ^ Handle to RecordIO object.
    , id `Ptr (Ptr CChar)'  -- ^ Pointer to return buffer.
    , alloca- `CSize' peek* -- ^ Size of buffer.
    } -> `Int' #}

-- | Set the current reader pointer position.
{#fun MXRecordIOReaderSeek as mxRecordIOReaderSeek
    { id `RecordIOHandle'   -- ^ Handle to RecordIO object.
    , id `CSize'            -- ^ Target position.
    } -> `Int' #}

-- | Create a MXRtc object.
{#fun MXRtcCreate as mxRtcCreate
    { `String'          -- ^ Name.
    , id `MXUInt'                   -- ^ Number of inputs.
    , id `MXUInt'                   -- ^ Number of outputs.
    , withStringArray* `[String]'   -- ^ Input names.
    , withStringArray* `[String]'   -- ^ Output names.
    , withNDArrayHandleArray* `[NDArrayHandle]'  -- ^ Inputs.
    , withNDArrayHandleArray* `[NDArrayHandle]'  -- ^ Outputs.
    , id `Ptr CChar'                -- ^ Kernel.
    , alloca- `RtcHandle' peek*     -- ^ The result RTC handle.
    } -> `Int' #}

-- | Run cuda kernel.
{#fun MXRtcPush as mxRtcPush
    { id `RtcHandle'                -- ^ Handle.
    , id `MXUInt'                   -- ^ Number of inputs.
    , id `MXUInt'                   -- ^ Number of outputs.
    , withNDArrayHandleArray* `[NDArrayHandle]'  -- ^ Inputs.
    , withNDArrayHandleArray* `[NDArrayHandle]'  -- ^ Outputs.
    , id `MXUInt'                   -- ^ Grid dim x
    , id `MXUInt'                   -- ^ Grid dim y
    , id `MXUInt'                   -- ^ Grid dim z
    , id `MXUInt'                   -- ^ Block dim x
    , id `MXUInt'                   -- ^ Block dim y
    , id `MXUInt'                   -- ^ Block dim z
    } -> `Int' #}

-- | Delete a MXRtc object.
{#fun MXRtcFree as mxRtcFree
    { id `RtcHandle'
    } -> `Int' #}

-- |
{#fun MXCustomOpRegister as mxCustomOpRegister
    { `String'                  -- ^ op type.
    , id `CustomOpPropCreator'
    } -> `Int' #}
