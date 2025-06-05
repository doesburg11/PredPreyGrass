using UnityEditor;
using UnityEngine;

using Codice.Client.Common.EventTracking;
using Codice.CM.Common;
using PlasticGui;
using Unity.PlasticSCM.Editor.UI;

namespace Unity.PlasticSCM.Editor.Views.PendingChanges.Dialogs
{
    internal class EmptyCommentDialog : PlasticDialog
    {
        internal bool UserChoseToNotDisplayWarningAgain { get; private set; }

        internal static bool ShouldContinueWithCheckin(
            EditorWindow parentWindow,
            WorkspaceInfo wkInfo)
        {
            var dialog = Create(
                wkInfo,
                PlasticLocalization.Name.NoCheckinCommentTitle.GetString(),
                PlasticLocalization.Name.NoCheckinCommentMessage.GetString(),
                PlasticLocalization.Name.SkipAndCheckin.GetString());

            // using the apply response as the 'Skip & Check in' button click
            if (dialog.RunModal(parentWindow) != ResponseType.Apply)
                return false;

            if (dialog.UserChoseToNotDisplayWarningAgain)
            {
                PlasticGuiConfig.Get().Configuration.ShowEmptyCommentWarning = false;
                PlasticGuiConfig.Get().Save();
            }

            return true;
        }

        internal static bool ShouldContinueWithShelve(
            EditorWindow parentWindow,
            WorkspaceInfo wkInfo)
        {
            var dialog = Create(
                wkInfo,
                PlasticLocalization.Name.NoShelveCommentTitle.GetString(),
                PlasticLocalization.Name.NoShelveCommentMessage.GetString(),
                PlasticLocalization.Name.SkipAndShelve.GetString());

            // using the apply response as the 'Skip & Shelve' button click
            if (dialog.RunModal(parentWindow) != ResponseType.Apply)
                return false;

            if (dialog.UserChoseToNotDisplayWarningAgain)
            {
                PlasticGuiConfig.Get().Configuration.ShowEmptyShelveCommentWarning = false;
                PlasticGuiConfig.Get().Save();
            }

            return true;
        }

        protected override string GetTitle()
        {
            return string.Empty;
        }

        protected override void OnModalGUI()
        {
            DoMainContentSection();

            DoCheckboxSection();

            DoButtonsArea();
        }

        void DoMainContentSection()
        {
            using (new EditorGUILayout.HorizontalScope())
            {

               DoIconArea();

                using (new EditorGUILayout.VerticalScope())
                {
                    GUILayout.Label(
                        mDialogTitle,
                        UnityStyles.Dialog.MessageTitle);

                    GUILayout.Space(3f);

                    GUILayout.Label(
                        mDialogMessage,
                        UnityStyles.Dialog.MessageText);

                    GUILayout.Space(15f);
                }
            }
        }

        void DoCheckboxSection()
        {
            using (new EditorGUILayout.HorizontalScope())
            {
                GUILayout.Space(70f);

                UserChoseToNotDisplayWarningAgain = TitleToggle(
                    PlasticLocalization.GetString(
                        PlasticLocalization.Name.DoNotShowMessageAgain),
                    UserChoseToNotDisplayWarningAgain);
            }
        }

        void DoButtonsArea()
        {
            using (new EditorGUILayout.VerticalScope())
            {
                GUILayout.Space(25f);

                using (new EditorGUILayout.HorizontalScope())
                {
                    GUILayout.FlexibleSpace();

                    if (Application.platform == RuntimePlatform.WindowsEditor)
                    {
                        DoAddCommentButton();
                        GUILayout.Space(13f);
                        DoSkipAndContinueButton();
                        return;
                    }

                    DoSkipAndContinueButton();
                    GUILayout.Space(13f);
                    DoAddCommentButton();
                }
            }
        }

        void DoSkipAndContinueButton()
        {
            if (!AcceptButton(mDialogNeutralButtonText,
                30))
                return;

            if (!mSentCheckinAnywayTrackEvent)
            {
                TrackFeatureUseEvent.For(
                  PlasticGui.Plastic.API.GetRepositorySpec(mWkInfo),
                  TrackFeatureUseEvent.Features.EmptyComment.PendingChangesCheckinDialogCheckinAnyway);

                mSentCheckinAnywayTrackEvent = true;
            }

            if (UserChoseToNotDisplayWarningAgain && !mSentCheckboxTrackEvent)
            {
                TrackFeatureUseEvent.For(
                    PlasticGui.Plastic.API.GetRepositorySpec(mWkInfo),
                    TrackFeatureUseEvent.Features.EmptyComment.PendingChangesCheckinDialogDoNotShowMessageAgain);

                mSentCheckboxTrackEvent = true;
            }

            ApplyButtonAction();
        }

        void DoAddCommentButton()
        {
            if (!NormalButton(PlasticLocalization.GetString(
                    PlasticLocalization.Name.AddComment)))
                return;

            if (!mSentCancelTrackEvent)
            {
                TrackFeatureUseEvent.For(
                    PlasticGui.Plastic.API.GetRepositorySpec(mWkInfo),
                    TrackFeatureUseEvent.Features.EmptyComment.PendingChangesCheckinDialogCancel);

                mSentCancelTrackEvent = true;
            }

            CancelButtonAction();
        }

        void DoIconArea()
        {
            GUILayout.BeginVertical();
            GUILayout.Space(10);

            Rect iconRect = GUILayoutUtility.GetRect(
                GUIContent.none, GUIStyle.none,
                GUILayout.Width(60), GUILayout.Height(60));
            iconRect.x -= 10;

            GUI.DrawTexture(
                iconRect,
                Images.GetPlasticIcon(),
                ScaleMode.ScaleToFit);

            GUILayout.EndVertical();
        }

        static EmptyCommentDialog Create(
            WorkspaceInfo wkInfo,
            string dialogTitle,
            string dialogMessage,
            string dialogNeutralButtonText)
        {
            var instance = CreateInstance<EmptyCommentDialog>();
            instance.mEnterKeyAction = instance.OkButtonAction;
            instance.mEscapeKeyAction = instance.CancelButtonAction;
            instance.mWkInfo = wkInfo;
            instance.mDialogTitle = dialogTitle;
            instance.mDialogMessage = dialogMessage;
            instance.mDialogNeutralButtonText = dialogNeutralButtonText;

            return instance;
        }

        WorkspaceInfo mWkInfo;
        string mDialogTitle;
        string mDialogMessage;
        string mDialogNeutralButtonText;

        // IMGUI evaluates every frame, need to make sure feature tracks get sent only once
        bool mSentCheckinAnywayTrackEvent = false;
        bool mSentCancelTrackEvent = false;
        bool mSentCheckboxTrackEvent = false;
    }
}
