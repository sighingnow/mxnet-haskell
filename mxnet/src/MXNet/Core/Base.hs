-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Base
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Interfaces in core module of MXNet.
--

module MXNet.Core.Base
    ( -- * Necessary raw functions
      mxGetLastError
    , mxListAllOpNames
      -- * NDArray
    , NDArray
    , waitAll
    , makeEmptyNDArray
    , makeNDArray
    , ndshape
    , ndsize
    , context
    , slice
    , at
    , waitToRead
    , onehotEncode
    , zeros
    , ones
    , full
    , array
      -- * Symbol
    , Symbol
    , variable
    , getName
    , getAttr
    , setAttr
    , infershape
    , grad
    , bind
    , bind'
    , listInputs
    , listOutputs
    , listAuxiliaries
      -- * Executor
    , Executor
    , makeExecutor
    , forward
    , backward
    , getOutput
      -- * DType
    , module MXNet.Core.Base.DType
      -- * Heterogeneous Dictionary.
    , module MXNet.Core.Base.HMap
    )where

import MXNet.Core.Base.DType
import MXNet.Core.Base.Executor
import MXNet.Core.Base.HMap
import MXNet.Core.Base.NDArray
import MXNet.Core.Base.Symbol
import MXNet.Core.Base.Internal
