import Distribution.Simple
import Distribution.Simple.Setup
import Distribution.Types.LocalBuildInfo
import Distribution.Types.GenericPackageDescription
import Distribution.Types.HookedBuildInfo

import Data.List (find)

main :: IO ()
main = defaultMainWithHooks $
    simpleUserHooks { confHook = confWithExtraFlags
                    }

-- | Patch the `-lmxnet` arguments during the build stage for TH phase.
confWithExtraFlags :: (GenericPackageDescription, HookedBuildInfo) -> ConfigFlags -> IO LocalBuildInfo
confWithExtraFlags desc_info flags =
    defaultConfHook desc_info newFlags
  where
    newFlags = flags { configProgramArgs = defaultOtherArgs ++ [ ("ghc", defaultGhcArgs ++ ["-lmxnet"])
                                                               , ("haddock", defaultHaddockArgs ++ ["--optghc=-lmxnet"])
                                                               ]
                     }
    defaultProgramArgs = configProgramArgs flags
    defaultOtherArgs = filter (\(prog, _) -> prog /= "ghc" && prog /= "haddock") defaultProgramArgs
    defaultGhcArgs = case find (\(prog, _) -> prog == "ghc") defaultProgramArgs of
                          Nothing -> []
                          Just (_, opts) -> opts
    defaultHaddockArgs = case find (\(prog, _) -> prog == "haddock") defaultProgramArgs of
                              Nothing -> []
                              Just (_, opts) -> opts
    defaultConfHook = confHook simpleUserHooks
