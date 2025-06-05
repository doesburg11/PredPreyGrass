using System.Collections;
using System.Reflection;

namespace UnityEditor.TestTools.TestRunner.TestRun.Tasks
{
    internal class SetInteractionModeTask : TestTaskBase
    {
        private const string ApplicationIdleTimeKey = "ApplicationIdleTime";
        private const string InteractionModeKey = "InteractionMode";
        public override IEnumerator Execute(TestJobData testJobData)
        {
#if UNITY_2020_3_OR_NEWER
            SetInteractionModeToNoThrottling(testJobData);
            EditorApplication.UpdateInteractionModeSettings();
#endif
            yield break;
        }
#if UNITY_2020_3_OR_NEWER
        private void SetInteractionModeToNoThrottling(TestJobData testJobData)
        {
            testJobData.UserApplicationIdleTime = EditorPrefs.GetInt(ApplicationIdleTimeKey);
            testJobData.UserInteractionMode = EditorPrefs.GetInt(InteractionModeKey);
            EditorPrefs.SetInt(ApplicationIdleTimeKey, 0);
            EditorPrefs.SetInt(InteractionModeKey, 1);
        }
#endif
    }
}