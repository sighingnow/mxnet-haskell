-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.NNVM.Internal.Raw
-- copyright:                   (c) 2016-2017 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Direct C FFI bindings for <nnvm/c_api.h>.
--
#if __GLASGOW_HASKELL__ >= 801
{-# LANGUAGE Strict #-}
#endif
{-# LANGUAGE ForeignFunctionInterface #-}

module MXNet.Core.NNVM.Internal.Raw where

import Control.Exception (throwIO)
import Foreign.C.String
import Foreign.C.Types
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Ptr
import Foreign.Storable

import C2HS.C.Extra.Marshal

{#import MXNet.Core.Types.Internal.Raw #}

#include <nnvm/c_api.h>

-- | Set the last error message needed by C API.
{#fun NNAPISetLastError as nnAPISetLastError
    { `String'
    } -> `()' #}

-- | Return str message of the last error.
{#fun NNGetLastError as nnGetLastError
    {
    } -> `String' #}

{#fun NNListAllOpNames as nnListAllOpNamesImpl
    { alloca- `NNUInt' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    } -> `Int' #}

-- | List all the available operator names, include entries.
nnListAllOpNames :: IO (Int, [String])
nnListAllOpNames = do
    (res, n, p) <- nnListAllOpNamesImpl
    ss <- peekStringArray n p
    return (res, ss)

-- | Get operator handle given name.
{#fun NNGetOpHandle as nnGetOpHandle
    { `String'                  -- ^ The name of the operator.
    , alloca- `OpHandle' peek*
    } -> `Int' #}

{#fun NNListUniqueOps as nnListUniqueOpsImpl
    { alloca- `NNUInt' peek*
    , alloca- `Ptr OpHandle' peek*
    } -> `Int' #}

-- | List all the available operators.
nnListUniqueOps :: IO (Int, [OpHandle])
nnListUniqueOps = do
    (res, n, p) <- nnListUniqueOpsImpl
    ops <- peekArray (fromIntegral n) p
    return (res, ops)

{#fun NNGetOpInfo as nnGetOpInfoImpl
    { id `OpHandle'
    , alloca- `String' peekString* -- ^ return a name (string).
    , alloca- `String' peekString* -- ^ return description (string).
    , alloca- `NNUInt' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    , alloca- `String' peekString*
    } -> `Int' #}

-- | Get the detailed information about atomic symbol.
nnGetOpInfo :: OpHandle
            -> IO (Int, String, String, NNUInt, [String], [String], [String], String)
nnGetOpInfo handle = do
    (res, name, desc, argc, pargv, pargt, pargdesc, rettype) <- nnGetOpInfoImpl handle
    argv <- peekStringArray argc pargv
    argt <- peekStringArray argc pargt
    argdesc <- peekStringArray argc pargdesc
    return (res, name, desc, argc, argv, argt, argdesc, rettype)

-- | Create an AtomicSymbol functor.
{#fun NNSymbolCreateAtomicSymbol as nnSymbolCreateAtomicSymbol
    { id `OpHandle'                 -- ^ The operator handle.
    , id `NNUInt'                   -- ^ The number of parameters.
    , withStringArray* `[String]'   -- ^ The keys to the params.
    , withStringArray* `[String]'   -- ^ The values to the params.
    , alloca- `SymbolHandle' peek*
    } -> `Int' #}

-- | Create a Variable Symbol.
{#fun NNSymbolCreateVariable as nnSymbolCreateVariable
    { `String'                      -- ^ The name of the variable.
    , alloca- `SymbolHandle' peek*
    } -> `Int' #}

-- | Create a Symbol by grouping list of symbols together.
{#fun NNSymbolCreateGroup as nnSymbolCreateGroup
    { id `NNUInt'                   -- ^ Number of symbols to be grouped.
    , withArray* `[SymbolHandle]'   -- ^ Array of symbol handles.
    , alloca- `SymbolHandle' peek*
    } -> `Int' #}

-- | Add src_dep to the handle as control dep.
{#fun NNAddControlDeps as nnAddControlDeps
    { id `SymbolHandle' -- ^ The symbol to add dependency edges on.
    , id `SymbolHandle' -- ^ The source handles.
    } -> `Int' #}

-- | Free the symbol handle.
{#fun NNSymbolFree as nnSymbolFree
    { id `SymbolHandle'
    } -> `Int' #}

-- | Copy the symbol to another handle.
{#fun NNSymbolCopy as nnSymbolCopy
    { id `SymbolHandle'
    , alloca- `SymbolHandle' peek*
    } -> `Int' #}

-- | Print the content of symbol, used for debug.
{#fun NNSymbolPrint as nnSymbolPrint
    { id `SymbolHandle'
    , alloca- `String' peekString*
    } -> `Int' #}

-- | Get string attribute from symbol.
{#fun NNSymbolGetAttr as nnSymbolGetAttr
    { id `SymbolHandle'             -- ^ symbol handle
    , `String'                      -- ^ key
    , alloca- `String' peekString*  -- ^ result value
    , alloca- `Int' peekIntegral*   -- ^ if success, 0 when success, -1 when failure happens.
    } -> `Int' #}

-- | Set string attribute from symbol.
{#fun NNSymbolSetAttrs as nnSymbolSetAttrs
    { id `SymbolHandle'
    , id `NNUInt'
    , withStringArray* `[String]' -- ^ attribute keys
    , withStringArray* `[String]' -- ^ attribute values
    } -> `Int' #}

{#fun NNSymbolListAttrs as nnSymbolListAttrsImpl
    { id `SymbolHandle'
    , `Int'                           -- ^ 0 for recursive, 1 for shallow
    , alloca- `NNUInt' peek*          -- ^ out size
    , alloca- `Ptr (Ptr CChar)' peek* -- ^ out attributes
    } -> `Int' #}

-- | Get all attributes from symbol, including all descendents.
nnSymbolListAttrs :: SymbolHandle -> Int -> IO (Int, [String])
nnSymbolListAttrs sym recursive = do
    (res, n, p) <- nnSymbolListAttrsImpl sym recursive
    ss <- peekStringArray n p
    return (res, ss)

{#fun NNSymbolListInputVariables as nnSymbolListInputVariablesImpl
    { id `SymbolHandle'
    , `Int'
    , alloca- `NNUInt' peek*
    , alloca- `Ptr SymbolHandle' peek*
    } -> `Int' #}

-- | List inputs variables in the symbol.
nnSymbolListInputVariables :: SymbolHandle -> Int -> IO (Int, [SymbolHandle])
nnSymbolListInputVariables sym opt = do
    (res, n, p) <- nnSymbolListInputVariablesImpl sym opt
    vs <- peekArray (fromIntegral n) p
    return (res, vs)

{#fun NNSymbolListInputNames as nnSymbolListInputNamesImpl
    { id `SymbolHandle'
    , `Int'
    , alloca- `NNUInt' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    } -> `Int' #}

-- | List input names in the symbol.
nnSymbolListInputNames :: SymbolHandle -> Int -> IO (Int, [String])
nnSymbolListInputNames sym opt = do
    (res, n, p) <- nnSymbolListInputNamesImpl sym opt
    ss <- peekStringArray n p
    return (res, ss)

{#fun NNSymbolListOutputNames as nnSymbolListOutputNamesImpl
    { id `SymbolHandle'
    , alloca- `NNUInt' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    } -> `Int' #}

-- | List returns names in the symbol.
nnSymbolListOutputNames :: SymbolHandle -> IO (Int, [String])
nnSymbolListOutputNames sym = do
    (res, n, p) <- nnSymbolListOutputNamesImpl sym
    ss <- peekStringArray n p
    return (res, ss)

-- | Get a symbol that contains all the internals.
{#fun NNSymbolGetInternals as nnSymbolGetInternals
    { id `SymbolHandle'
    , alloca- `SymbolHandle' peek*
    } -> `Int' #}

-- | Get index-th outputs of the symbol.
{#fun NNSymbolGetOutput as nnSymbolGetOutput
    { id `SymbolHandle'
    , id `NNUInt'
    , alloca- `SymbolHandle' peek*
    } -> `Int' #}

-- | Compose the symbol on other symbols.
{#fun NNSymbolCompose as nnSymbolComposeImpl
    { id `SymbolHandle'
    , id `Ptr CChar'
    , id `NNUInt'
    , withStringArray* `[String]'
    , withArray* `[SymbolHandle]'
    } -> `Int' #}


-- | Invoke a nnvm op and imperative function.
nnSymbolCompose :: SymbolHandle       -- ^ Creator/Handler of the OP.
                -> String
                -> [String]
                -> [SymbolHandle]
                -> IO Int
nnSymbolCompose sym name keys args = do
    if null keys || length keys == length args
        then return ()
        else throwIO $ userError "nnSymbolCompose: keyword arguments mismatch."
    let nargs = fromIntegral $ length args
    if null name
        then nnSymbolComposeImpl sym nullPtr nargs keys args
        else withCString name $ \p -> nnSymbolComposeImpl sym p nargs keys args

-- | Create a graph handle from symbol.
{#fun NNGraphCreate as nnGraphCreate
    { id `SymbolHandle'
    , alloca- `GraphHandle' peek*
    } -> `Int' #}

-- | Free the graph handle.
{#fun NNGraphFree as nnGraphFree
    { id `GraphHandle'
    } -> `Int' #}

-- | Get a new symbol from the graph.
{#fun NNGraphGetSymbol as nnGraphGetSymbol
    { id `GraphHandle'             -- ^ the graph handle.
    , alloca- `SymbolHandle' peek* -- ^ the corresponding symbol.
    } -> `Int' #}

-- | Get Set a attribute in json format.
{#fun NNGraphSetJSONAttr as nnGraphSetJSONAttr
    { id `GraphHandle'
    , `String'
    , `String'
    } -> `Int' #}

-- | Get a serialized attrirbute from graph.
{#fun NNGraphGetJSONAttr as nnGraphGetJSONAttr
    { id `SymbolHandle'
    , `String'
    , alloca- `String' peekString*
    , alloca- `Int' peekIntegral* -- ^ if success.
    } -> `Int' #}

-- | Set a attribute whose type is std::vector<NodeEntry> in c++.
{#fun NNGraphSetNodeEntryListAttr_ as nnGraphSetNodeEntryListAttr_
    { id `GraphHandle'
    , `String'
    , `SymbolHandle'
    } -> `Int' #}

-- | Apply passes on the src graph.
{#fun NNGraphApplyPasses as nnGraphApplyPasses
    { id `GraphHandle'
    , id `NNUInt'
    , withStringArray* `[String]'
    , alloca- `GraphHandle' peek*
    } -> `Int' #}
