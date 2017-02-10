-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.NNVM.Base.Internal.Raw
-- copyright:                   (c) 2016-2017 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Direct C FFI bindings for <nnvm/c_api.h>.
--
#if __GLASGOW_HASKELL__ >= 709
{-# LANGUAGE Safe #-}
#elif __GLASGOW_HASKELL__ >= 701
{-# LANGUAGE Trustworthy #-}
#endif
{-# LANGUAGE ForeignFunctionInterface #-}

module MXNet.Core.NNVM.Base.Internal.Raw where

import Foreign.C.String
import Foreign.C.Types
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Marshal.Utils ( with )
import Foreign.Ptr
import Foreign.Storable

import C2HS.C.Extra.Marshal

{#import MXNet.Core.Internal.Types.Raw #}

#include <nnvm/c_api.h>

{#fun NNAPISetLastError as nnAPISetLastError
    { `String'
    } -> `()' #}

{#fun NNGetLastError as nnGetLastError
    {
    } -> `String' #}

{#fun NNListAllOpNames as nnListAllOpNamesImpl
    { alloca- `NNUInt' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    } -> `Int' #}

nnListAllOpNames :: IO (Int, NNUInt, [String])
nnListAllOpNames = do
    (res, n, p) <- nnListAllOpNamesImpl
    ss <- peekStringArray n p
    return (res, n, ss)

{#fun NNGetOpHandle as nnGetOpHandle
    { `String'
    , alloca- `OpHandle' peek*
    } -> `Int' #}

{#fun NNListUniqueOps as nnListUniqueOpsImpl
    { alloca- `NNUInt' peek*
    , alloca- `Ptr OpHandle' peek*
    } -> `Int' #}

nnListUniqueOps :: IO (Int, NNUInt, [OpHandle])
nnListUniqueOps = do
    (res, n, p) <- nnListUniqueOpsImpl
    ops <- peekArray (fromIntegral n) p
    return (res, n, ops)

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

nnGetOpInfo :: OpHandle
            -> IO (Int, String, String, NNUInt, [String], [String], [String], String)
nnGetOpInfo handle = do
    (res, name, desc, argc, pargv, pargt, pargdesc, rettype) <- nnGetOpInfoImpl handle
    argv <- peekStringArray argc pargv
    argt <- peekStringArray argc pargt
    argdesc <- peekStringArray argc pargdesc
    return (res, name, desc, argc, argv, argt, argdesc, rettype)

{#fun NNSymbolCreateAtomicSymbol as nnSymbolCreateAtomicSymbol
    { id `OpHandle'
    , id `NNUInt'
    , withStringArray* `[String]'
    , withStringArray* `[String]'
    , alloca- `SymbolHandle' peek*
    } -> `Int' #}

{#fun NNSymbolCreateVariable as nnSymbolCreateVariable
    { `String'
    , alloca- `SymbolHandle' peek*
    } -> `Int' #}

{#fun NNSymbolCreateGroup as nnSymbolCreateGroup
    { id `NNUInt'
    , withArray* `[SymbolHandle]'
    , alloca- `SymbolHandle' peek*
    } -> `Int' #}

{#fun NNAddControlDeps as nnAddControlDeps
    { id `SymbolHandle' -- ^ The symbol to add dependency edges on.
    , id `SymbolHandle' -- ^ The source handles.
    } -> `Int' #}

{#fun NNSymbolFree as nnSymbolFree
    { id `SymbolHandle'
    } -> `Int' #}

{#fun NNSymbolCopy as nnSymbolCopy
    { id `SymbolHandle'
    , alloca- `SymbolHandle' peek*
    } -> `Int' #}

{#fun NNSymbolPrint as nnSymbolPrint
    { id `SymbolHandle'
    , alloca- `String' peekString*
    } -> `Int' #}

{#fun NNSymbolGetAttr as nnSymbolGetAttr
    { id `SymbolHandle'             -- ^ symbol handle
    , `String'                      -- ^ key
    , alloca- `String' peekString*  -- ^ result value
    , alloca- `Int' peekIntegral*   -- ^ if success, 0 when success, -1 when failure happens.
    } -> `Int' #}

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

nnSymbolListAttrs :: SymbolHandle -> Int -> IO (Int, NNUInt, [String])
nnSymbolListAttrs sym recursive = do
    (res, n, p) <- nnSymbolListAttrsImpl sym recursive
    ss <- peekStringArray n p
    return (res, n, ss)

{#fun NNSymbolListInputVariables as nnSymbolListInputVariablesImpl
    { id `SymbolHandle'
    , `Int'
    , alloca- `NNUInt' peek*
    , alloca- `Ptr SymbolHandle' peek*
    } -> `Int' #}

nnSymbolListInputVariables :: SymbolHandle -> Int -> IO (Int, NNUInt, [SymbolHandle])
nnSymbolListInputVariables sym opt = do
    (res, n, p) <- nnSymbolListInputVariablesImpl sym opt
    vs <- peekArray (fromIntegral n) p
    return (res, n, vs)

{#fun NNSymbolListInputNames as nnSymbolListInputNamesImpl
    { id `SymbolHandle'
    , `Int'
    , alloca- `NNUInt' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    } -> `Int' #}

nnSymbolListInputNames :: SymbolHandle -> Int -> IO (Int, NNUInt, [String])
nnSymbolListInputNames sym opt = do
    (res, n, p) <- nnSymbolListInputNamesImpl sym opt
    ss <- peekStringArray n p
    return (res, n, ss)

{#fun NNSymbolListOutputNames as nnSymbolListOutputNamesImpl
    { id `SymbolHandle'
    , alloca- `NNUInt' peek*
    , alloca- `Ptr (Ptr CChar)' peek*
    } -> `Int' #}

nnSymbolListOutputNames :: SymbolHandle -> IO (Int, NNUInt, [String])
nnSymbolListOutputNames sym = do
    (res, n, p) <- nnSymbolListOutputNamesImpl sym
    ss <- peekStringArray n p
    return (res, n, ss)

{#fun NNSymbolGetInternals as nnSymbolGetInternals
    { id `SymbolHandle'
    , alloca- `SymbolHandle' peek*
    } -> `Int' #}

{#fun NNSymbolGetOutput as nnSymbolGetOutput
    { id `SymbolHandle'
    , id `NNUInt'
    , alloca- `SymbolHandle' peek*
    } -> `Int' #}

{#fun NNSymbolCompose as nnSymbolCompose
    { id `SymbolHandle'
    , `String'
    , id `NNUInt'
    , withStringArray* `[String]'
    , alloca- `SymbolHandle' peek*
    } -> `Int' #}

{#fun NNGraphCreate as nnGraphCreate
    { id `SymbolHandle'
    , alloca- `GraphHandle' peek*
    } -> `Int' #}

{#fun NNGraphFree as nnGraphFree
    { id `GraphHandle'
    } -> `Int' #}

{#fun NNGraphGetSymbol as nnGraphGetSymbol
    { id `GraphHandle'             -- ^ the graph handle.
    , alloca- `SymbolHandle' peek* -- ^ the corresponding symbol.
    } -> `Int' #}

{#fun NNGraphSetJSONAttr as nnGraphSetJSONAttr
    { id `GraphHandle'
    , `String'
    , `String'
    } -> `Int' #}

{#fun NNGraphGetJSONAttr as nnGraphGetJSONAttr
    { id `SymbolHandle'
    , `String'
    , alloca- `String' peekString*
    , alloca- `Int' peekIntegral* -- ^ if success.
    } -> `Int' #}

{#fun NNGraphSetNodeEntryListAttr_ as nnGraphSetNodeEntryListAttr_
    { id `GraphHandle'
    , `String'
    , `SymbolHandle'
    } -> `Int' #}

{#fun NNGraphApplyPasses as nnGraphApplyPasses
    { id `GraphHandle'
    , id `NNUInt'
    , withStringArray* `[String]'
    , alloca- `GraphHandle' peek*
    } -> `Int' #}
