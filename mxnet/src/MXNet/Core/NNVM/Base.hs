-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.NNVM.Base
-- copyright:                   (c) 2016-2017 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Interfaces in core module of NNVM.
--
module MXNet.Core.NNVM.Base (
      -- * Re-export data type definitions
      NNUInt
    , OpHandle
    , SymbolHandle
    , GraphHandle
      -- * Re-export functions.
    , nnAPISetLastError
    , nnGetLastError
    , nnListAllOpNames
    , nnGetOpHandle
    , nnListUniqueOps
    , nnGetOpInfo
    , nnSymbolCreateAtomicSymbol
    , nnSymbolCreateVariable
    , nnSymbolCreateGroup
    , nnAddControlDeps
    , nnSymbolFree
    , nnSymbolCopy
    , nnSymbolPrint
    , nnSymbolGetAttr
    , nnSymbolSetAttrs
    , nnSymbolListAttrs
    , nnSymbolListInputVariables
    , nnSymbolListInputNames
    , nnSymbolListOutputNames
    , nnSymbolGetInternals
    , nnSymbolGetOutput
    , nnSymbolCompose
    , nnGraphCreate
    , nnGraphFree
    , nnGraphGetSymbol
    , nnGraphSetJSONAttr
    , nnGraphGetJSONAttr
    , nnGraphSetNodeEntryListAttr_
    , nnGraphApplyPasses
    ) where

import MXNet.Core.Internal.Types.Raw
import MXNet.Core.NNVM.Base.Internal.Raw
