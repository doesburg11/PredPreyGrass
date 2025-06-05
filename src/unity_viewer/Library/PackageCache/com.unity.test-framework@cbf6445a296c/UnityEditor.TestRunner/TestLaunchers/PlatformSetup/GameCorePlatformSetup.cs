using System;
using UnityEditor.Build;

#if UNITY_GAMECORE
using UnityEditor.GameCore;
#endif

namespace UnityEditor.TestTools.TestRunner
{
    internal class GameCorePlatformSetup : IPlatformSetup
    {
#if UNITY_GAMECORE
        private const GameCoreDeployMethod k_deployMethod = GameCoreDeployMethod.Push;
        private const string k_PlsSize = "512";
        private const string k_PlsGrow = "1024";
        private const string k_TitleID = "73ECA03C";
        private const string k_MSAAppId = "0000000040283475";
        private const string k_SCID = "00000000-0000-0000-0000-000073ECA03C";

        private static GameCoreSettings GetGameCoreSettings()
        {
#if UNITY_GAMECORE_XBOXSERIES
            return GameCoreScarlettSettings.GetInstance();
#elif UNITY_GAMECORE_XBOXONE
            return GameCoreXboxOneSettings.GetInstance();
#endif
        }
#endif

        public void Setup()
        {
#if UNITY_GAMECORE
            GameCoreSettings settings = GetGameCoreSettings();

            if (settings == null)
                return;

            settings.DeploymentMethod = k_deployMethod;

            settings.SCID = k_SCID;
            settings.GameConfig.MSAAppId = k_MSAAppId;
            settings.GameConfig.TitleId = k_TitleID;

            settings.GameConfig.PersistentLocalStorage = new GameCore.GameConfig.CT_PersistentLocalStorage();
            settings.GameConfig.PersistentLocalStorage.SizeMB = k_PlsSize;
            settings.GameConfig.PersistentLocalStorage.GrowableToMB = k_PlsGrow;
            settings.SaveGameConfig();

            NamedBuildTarget namedTarget = NamedBuildTarget.FromActiveSettings(EditorUserBuildSettings.activeBuildTarget);
            PlayerSettings.productName = $"Test_{namedTarget}";
#endif
        }

        public void PostBuildAction()
        {
        }

        public void PostSuccessfulBuildAction()
        {
        }

        public void PostSuccessfulLaunchAction()
        {
        }

        public void CleanUp()
        {

        }
    }
}
