-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Symbol
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Symbol module.
--
module MXNet.Core.Symbol where

{-
      -- (
      -- * Functions about NDArray
      -- * Default contexts
      -- ) where
-}

import           Control.Monad

import           MXNet.Core.Base

-- | Type alias for variable.
type Var = SymbolHandle

-- | Make a new symbolic variable with given name.
makeVar :: String   -- ^ Name.
        -> IO Var   -- ^ Result variable.
makeVar = mxSymbolCreateVariable >=> return . snd

-- | Get the name of a given variable.
getVarName :: Var -> IO String
getVarName = mxSymbolGetName >=> \(_, name, _) -> return name

