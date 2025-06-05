using System;

using UnityEditor;
using UnityEngine;

using Codice.Client.Common;
using PlasticGui;
using PlasticGui.WorkspaceWindow.PendingChanges;
using Unity.PlasticSCM.Editor.UI;

namespace Unity.PlasticSCM.Editor.Settings
{
    class ShelveAndSwitchOptionsFoldout
    {
        internal void OnActivate()
        {
            PlasticGuiConfigData data = PlasticGuiConfig.Get().Configuration;

            mUndoChangesAfterShelve = data.ShelvingUndoBehavior == UndoShelveBehavior.UndoChanges;
            mKeepChangesAfterShelve = data.ShelvingUndoBehavior == UndoShelveBehavior.KeepChanges;
            mAskAlwaysAfterShelve = data.ShelvingUndoBehavior == UndoShelveBehavior.Ask;

            mSwitchActionShelveOption  = ClientConfig.Get().GetPendingChangesOnSwitchAction() == UserAction.Shelve;
            mSwitchActionAllowOption = ClientConfig.Get().GetPendingChangesOnSwitchAction() == UserAction.None;
            mSwitchActionWarnOption = ClientConfig.Get().GetPendingChangesOnSwitchAction() == UserAction.Warn;
            mSwitchActionFailOption = ClientConfig.Get().GetPendingChangesOnSwitchAction() == UserAction.Fail ||
                                      ClientConfig.Get().GetPendingChangesOnSwitchAction() == UserAction.NotDefined;
        }

        internal void OnDeactivate()
        {
            PlasticGuiConfig plasticGuiConfig = PlasticGuiConfig.Get();
            plasticGuiConfig.Configuration.ShelvingUndoBehavior = GetSelectedShelvingUndoBehavior();
            plasticGuiConfig.Save();

            ClientConfigData configData = ClientConfig.Get().GetClientConfigData();
            configData.SetPendingChangesOnSwitchAction(GetSelectedSwitchBehavior());
            ClientConfig.Get().Save(configData);
        }

        internal void OnGUI()
        {
            DrawSplitter.ForWidth(UnityConstants.SETTINGS_GUI_WIDTH);

            DrawSettingsSection(DoDiffAndMergePreferences);
        }

        internal void SelectShelvingUndoBehaviorForTesting(UndoShelveBehavior shelveBehavior)
        {
            mUndoChangesAfterShelve = shelveBehavior == UndoShelveBehavior.UndoChanges;
            mKeepChangesAfterShelve = shelveBehavior == UndoShelveBehavior.KeepChanges;
            mAskAlwaysAfterShelve = shelveBehavior == UndoShelveBehavior.Ask;
        }

        internal void SelectSwitchActionForTesting(UserAction switchAction)
        {
            mSwitchActionShelveOption  = switchAction == UserAction.Shelve;
            mSwitchActionAllowOption = switchAction == UserAction.None;
            mSwitchActionWarnOption = switchAction == UserAction.Warn;
            mSwitchActionFailOption = switchAction == UserAction.Fail ||
                                      switchAction == UserAction.NotDefined;
        }

        UndoShelveBehavior GetSelectedShelvingUndoBehavior()
        {
            if (mUndoChangesAfterShelve)
                return UndoShelveBehavior.UndoChanges;

            if (mKeepChangesAfterShelve)
                return UndoShelveBehavior.KeepChanges;

            return UndoShelveBehavior.Ask;
        }

        UserAction GetSelectedSwitchBehavior()
        {
            if (mSwitchActionAllowOption)
                return UserAction.None;

            if (mSwitchActionWarnOption)
                return UserAction.Warn;

            if (mSwitchActionFailOption)
                return UserAction.Fail;

            return UserAction.Shelve;
        }

        void DoDiffAndMergePreferences()
        {
            DoUndoAfterShelveBehaviorSettings();

            DoSwitchActionSettings();
        }

        void DoUndoAfterShelveBehaviorSettings()
        {
            GUILayout.Label(
                PlasticLocalization.Name.UndoChangesAfterShelveDefaultBehavior.GetString(),
                UnityStyles.ProjectSettings.SectionTitle);
            EditorGUILayout.Space(2);

            if (EditorGUILayout.Toggle(
                    Styles.UndoChangesAfterShelve,
                    mUndoChangesAfterShelve,
                    new GUIStyle(EditorStyles.radioButton)))
            {
                mUndoChangesAfterShelve = true;
                mKeepChangesAfterShelve = false;
                mAskAlwaysAfterShelve = false;
            }

            if (EditorGUILayout.Toggle(
                    Styles.KeepChangesAfterShelve,
                    mKeepChangesAfterShelve,
                    new GUIStyle(EditorStyles.radioButton)))
            {
                mUndoChangesAfterShelve = false;
                mKeepChangesAfterShelve = true;
                mAskAlwaysAfterShelve = false;
            }

            if (EditorGUILayout.Toggle(
                    Styles.AskAlwaysAfterShelve,
                    mAskAlwaysAfterShelve,
                    new GUIStyle(EditorStyles.radioButton)))
            {
                mUndoChangesAfterShelve = false;
                mKeepChangesAfterShelve = false;
                mAskAlwaysAfterShelve = true;
            }
        }

        void DoSwitchActionSettings()
        {
            EditorGUILayout.Space(10);
            GUILayout.Label(
                PlasticLocalization.Name.SwitchActionOptionsTitle.GetString(),
                UnityStyles.ProjectSettings.SectionTitle);
            EditorGUILayout.Space(2);

            if (EditorGUILayout.Toggle(
                    Styles.SwitchActionShelve,
                    mSwitchActionShelveOption,
                    new GUIStyle(EditorStyles.radioButton)))
            {
                mSwitchActionShelveOption = true;
                mSwitchActionAllowOption = false;
                mSwitchActionWarnOption = false;
                mSwitchActionFailOption = false;
            }

            if (EditorGUILayout.Toggle(
                    Styles.SwitchActionAllow,
                    mSwitchActionAllowOption,
                    new GUIStyle(EditorStyles.radioButton)))
            {
                mSwitchActionShelveOption = false;
                mSwitchActionAllowOption = true;
                mSwitchActionWarnOption = false;
                mSwitchActionFailOption = false;
            }

            if (EditorGUILayout.Toggle(
                    Styles.SwitchActionWarn,
                    mSwitchActionWarnOption,
                    new GUIStyle(EditorStyles.radioButton)))
            {
                mSwitchActionShelveOption = false;
                mSwitchActionAllowOption = false;
                mSwitchActionWarnOption = true;
                mSwitchActionFailOption = false;
            }

            if (EditorGUILayout.Toggle(
                    Styles.SwitchActionFail,
                    mSwitchActionFailOption,
                    new GUIStyle(EditorStyles.radioButton)))
            {
                mSwitchActionShelveOption = false;
                mSwitchActionAllowOption = false;
                mSwitchActionWarnOption = false;
                mSwitchActionFailOption = true;
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
            internal static GUIContent UndoChangesAfterShelve =
                new GUIContent(
                    PlasticLocalization.Name.UndoChangesAfterShelve.GetString(),
                    PlasticLocalization.Name.UndoChangesAfterShelveExplanation.GetString());

            internal static GUIContent KeepChangesAfterShelve =
                new GUIContent(
                    PlasticLocalization.Name.KeepChangesAfterShelve.GetString(),
                    PlasticLocalization.Name.KeepChangesAfterShelveExplanation.GetString());

            internal static GUIContent AskAlwaysAfterShelve =
                new GUIContent(
                    PlasticLocalization.Name.AskAlwaysAfterShelve.GetString(),
                    PlasticLocalization.Name.AskAlwaysAfterShelveExplanation.GetString());

            internal static GUIContent SwitchActionShelve =
                new GUIContent(
                    PlasticLocalization.Name.SwitchActionShelveOption.GetString(),
                    PlasticLocalization.Name.SwitchActionShelveExplanation.GetString());

            internal static GUIContent SwitchActionAllow =
                new GUIContent(
                    PlasticLocalization.Name.SwitchActionAllowOption.GetString(),
                    PlasticLocalization.Name.SwitchActionAllowExplanation.GetString());

            internal static GUIContent SwitchActionWarn =
                new GUIContent(
                    PlasticLocalization.Name.SwitchActionWarnOption.GetString(),
                    PlasticLocalization.Name.SwitchActionWarnExplanation.GetString());

            internal static GUIContent SwitchActionFail =
                new GUIContent(
                    PlasticLocalization.Name.SwitchActionFailOption.GetString(),
                    PlasticLocalization.Name.SwitchActionFailExplanation.GetString());
        }

        bool mUndoChangesAfterShelve;
        bool mKeepChangesAfterShelve;
        bool mAskAlwaysAfterShelve;

        bool mSwitchActionShelveOption;
        bool mSwitchActionAllowOption;
        bool mSwitchActionWarnOption;
        bool mSwitchActionFailOption;
    }
}
