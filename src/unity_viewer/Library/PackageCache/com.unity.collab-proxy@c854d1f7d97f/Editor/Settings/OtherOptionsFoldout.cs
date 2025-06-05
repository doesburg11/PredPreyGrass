using System;

using UnityEditor;
using UnityEngine;

using PlasticGui;
using Unity.PlasticSCM.Editor.UI;

namespace Unity.PlasticSCM.Editor.Settings
{
    class OtherOptionsFoldout
    {
        internal void OnActivate()
        {
            PlasticGuiConfigData data = PlasticGuiConfig.Get().Configuration;

            mNewCodeReviewCreateAndOpenInDesktop = data.NewCodeReviewBehavior == NewCodeReviewBehavior.CreateAndOpenInDesktop;
            mNewCodeReviewRequestReviewInUnityCloud = data.NewCodeReviewBehavior == NewCodeReviewBehavior.RequestFromUnityCloud;
            mNewCodeReviewAskAlways = data.NewCodeReviewBehavior == NewCodeReviewBehavior.Ask;
        }

        internal void OnDeactivate()
        {
            PlasticGuiConfig plasticGuiConfig = PlasticGuiConfig.Get();
            plasticGuiConfig.Configuration.NewCodeReviewBehavior = GetSelectedNewCodeReviewBehavior();
            plasticGuiConfig.Save();
        }

        internal void OnGUI()
        {
            DrawSplitter.ForWidth(UnityConstants.SETTINGS_GUI_WIDTH);

            DrawSettingsSection(DoNewCodeReviewBehaviorSettings);
        }

        internal void SelectNewCodeReviewBehaviorForTesting(NewCodeReviewBehavior shelveBehavior)
        {
            mNewCodeReviewCreateAndOpenInDesktop = shelveBehavior == NewCodeReviewBehavior.CreateAndOpenInDesktop;
            mNewCodeReviewRequestReviewInUnityCloud = shelveBehavior == NewCodeReviewBehavior.RequestFromUnityCloud;
            mNewCodeReviewAskAlways = shelveBehavior == NewCodeReviewBehavior.Ask;
        }

        NewCodeReviewBehavior GetSelectedNewCodeReviewBehavior()
        {
            if (mNewCodeReviewCreateAndOpenInDesktop)
                return NewCodeReviewBehavior.CreateAndOpenInDesktop;

            if (mNewCodeReviewRequestReviewInUnityCloud)
                return NewCodeReviewBehavior.RequestFromUnityCloud;

            return NewCodeReviewBehavior.Ask;
        }

        void DoNewCodeReviewBehaviorSettings()
        {
            GUILayout.Label(
                PlasticLocalization.Name.NewCodeReviewDefaultBehavior.GetString(),
                UnityStyles.ProjectSettings.SectionTitle);
            EditorGUILayout.Space(2);

            if (EditorGUILayout.Toggle(
                    Styles.NewCodeReviewCreateAndOpenInDesktop,
                    mNewCodeReviewCreateAndOpenInDesktop,
                    new GUIStyle(EditorStyles.radioButton)))
            {
                mNewCodeReviewCreateAndOpenInDesktop = true;
                mNewCodeReviewRequestReviewInUnityCloud = false;
                mNewCodeReviewAskAlways = false;
            }

            if (EditorGUILayout.Toggle(
                    Styles.NewCodeReviewRequestReviewInUnityCloud,
                    mNewCodeReviewRequestReviewInUnityCloud,
                    new GUIStyle(EditorStyles.radioButton)))
            {
                mNewCodeReviewCreateAndOpenInDesktop = false;
                mNewCodeReviewRequestReviewInUnityCloud = true;
                mNewCodeReviewAskAlways = false;
            }

            if (EditorGUILayout.Toggle(
                    Styles.NewCodeReviewAskAlways,
                    mNewCodeReviewAskAlways,
                    new GUIStyle(EditorStyles.radioButton)))
            {
                mNewCodeReviewCreateAndOpenInDesktop = false;
                mNewCodeReviewRequestReviewInUnityCloud = false;
                mNewCodeReviewAskAlways = true;
            }
        }

        static void DrawSettingsSection(Action drawSettings)
        {
            float originalLabelWidth = EditorGUIUtility.labelWidth;

            try
            {
                EditorGUIUtility.labelWidth = UnityConstants.SETTINGS_GUI_WIDTH;

                using (new EditorGUILayout.HorizontalScope())
                {
                    GUILayout.Space(10);

                    using (new EditorGUILayout.VerticalScope())
                    {
                        GUILayout.Space(10);

                        drawSettings();

                        GUILayout.Space(10);
                    }

                    GUILayout.Space(10);
                }
            }
            finally
            {
                EditorGUIUtility.labelWidth = originalLabelWidth;
            }
        }

        class Styles
        {
            internal static GUIContent NewCodeReviewAskAlways =
                new GUIContent(
                    PlasticLocalization.Name.NewCodeReviewAskAlways.GetString(),
                    PlasticLocalization.Name.NewCodeReviewAskAlwaysExplanation.GetString());

            internal static GUIContent NewCodeReviewCreateAndOpenInDesktop =
                new GUIContent(
                    PlasticLocalization.Name.OpenInDesktopApp.GetString(),
                    PlasticLocalization.Name.CreateAndOpenCodeReviewInDesktopExplanation.GetString());

            internal static GUIContent NewCodeReviewRequestReviewInUnityCloud =
                new GUIContent(
                    PlasticLocalization.Name.OpenInUnityCloud.GetString(),
                    PlasticLocalization.Name.RequestCodeReviewFromUnityCloudExplanation.GetString());
        }

        bool mNewCodeReviewCreateAndOpenInDesktop;
        bool mNewCodeReviewRequestReviewInUnityCloud;
        bool mNewCodeReviewAskAlways;
    }
}
