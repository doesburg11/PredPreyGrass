using System;

using UnityEditor;
using UnityEngine;

using Codice.Client.Commands;
using Codice.Client.Common;
using PlasticGui;
using Unity.PlasticSCM.Editor.UI;

namespace Unity.PlasticSCM.Editor.Views
{
    internal class ShelvePendingChangesQuestionerBuilder :
        IShelvePendingChangesQuestionerBuilder
    {
        internal ShelvePendingChangesQuestionerBuilder(EditorWindow window)
        {
            mWindow = window;
        }

        public IShelvePendingChangesQuestioner Get()
        {
            return new ShelvePendingChangesQuestioner(mWindow);
        }

        readonly EditorWindow mWindow;
    }

    internal class ShelvePendingChangesQuestioner : IShelvePendingChangesQuestioner
    {
        internal ShelvePendingChangesQuestioner(EditorWindow window)
        {
            mWindow = window;
        }

        ShelvePendingChangesAction IShelvePendingChangesQuestioner.ShelvePendingChanges(
            string srcObject,
            string dstObject,
            bool canBringPendingChanges,
            bool warnOverwriteExistingShelves)
        {
            ShelvePendingChangesAction result = ShelvePendingChangesAction.Cancel;

            GUIActionRunner.RunGUIAction(() =>
            {
                result = ShelvePendingChangesDialog.ConfirmContinue(
                    mWindow,
                    srcObject,
                    dstObject,
                    canBringPendingChanges,
                    warnOverwriteExistingShelves);
            });

            return result;
        }

        readonly EditorWindow mWindow;
    }

    internal class ShelvePendingChangesDialog : PlasticDialog
    {
        protected override Rect DefaultRect
        {
            get
            {
                var baseRect = base.DefaultRect;
                return new Rect(baseRect.x, baseRect.y, 600, 300);
            }
        }

        internal static ShelvePendingChangesAction ConfirmContinue(
            EditorWindow window,
            string srcObject,
            string dstObject,
            bool canBringPendingChanges,
            bool warnOverwriteExistingShelves)
        {
            ShelvePendingChangesDialog dialog = Create(
                srcObject,
                dstObject,
                canBringPendingChanges,
                warnOverwriteExistingShelves);

            ResponseType result = dialog.RunModal(window);

            if (result == ResponseType.Ok)
                return ShelvePendingChangesAction.BringPendingChanges;

            if (result == ResponseType.None)
                return ShelvePendingChangesAction.ShelvePendingChanges;

            return ShelvePendingChangesAction.Cancel;
        }

        static ShelvePendingChangesDialog Create(
            string srcObject,
            string dstObject,
            bool canBringPendingChanges,
            bool warnOverwriteExistingShelves)
        {
            var instance = CreateInstance<ShelvePendingChangesDialog>();
            instance.mEnterKeyAction = instance.DoSwitchAction;
            instance.mEscapeKeyAction = instance.CancelButtonAction;
            instance.mSrcObject = srcObject;
            instance.mDstObject = dstObject;
            instance.mCanBringPendingChanges = canBringPendingChanges;
            instance.mWarnOverwriteExistingShelves = warnOverwriteExistingShelves;
            return instance;
        }

        protected override string GetTitle()
        {
            return PlasticLocalization.Name.ShelveAndSwitchDialogTitle.GetString();
        }

        bool AskForOverwriteConfirmation()
        {
            return GuiMessage.ShowQuestion(
                PlasticLocalization.Name.OverwriteShelveDialogTitle.GetString(),
                PlasticLocalization.Name.OverwriteShelveDialogMessage.GetString(),
                PlasticLocalization.Name.OverwriteShelveDialogOverwriteButton.GetString(),
                PlasticLocalization.Name.CancelButton.GetString(),
                null) == GuiMessage.GuiMessageResponseButton.Positive;
        }

        protected override void OnModalGUI()
        {
            Title(PlasticLocalization.Name.ShelveAndSwitchDialogTitle.GetString());

            Paragraph(PlasticLocalization.Name.ShelveAndSwitchExplanation.GetString());

            DoRadioButtonsArea();

            GUILayout.Space(20);

            DoButtonsArea();
        }

        void DoRadioButtonsArea()
        {
            if (GUILayout.Toggle(
                    mLeaveRadioToggle,
                    PlasticLocalization.Name.LeaveChangesButton.GetString(mSrcObject),
                    UnityStyles.Dialog.BoldRadioToggle))
            {
                mLeaveRadioToggle = true;
                mBringRadioToggle = false;
            }

            DrawTabbedContent(DrawLeaveRadioToggleExplanation);

            GUILayout.Space(10);

            GUI.enabled = mCanBringPendingChanges;

            if (GUILayout.Toggle(
                    mBringRadioToggle,
                    PlasticLocalization.Name.BringChangesButton.GetString(mDstObject),
                    UnityStyles.Dialog.BoldRadioToggle))
            {
                mLeaveRadioToggle = false;
                mBringRadioToggle = true;
            }

            DrawTabbedContent(DrawBringRadioToggleExplanation);

            GUI.enabled = true;

            GUILayout.FlexibleSpace();

            if (!mWarnOverwriteExistingShelves || !mLeaveRadioToggle)
                return;

            GUILayout.Label(
                PlasticLocalization.Name.OverwriteExistingShelvesWarning.GetString(),
                UnityStyles.Dialog.BoldText);
        }

        void DrawLeaveRadioToggleExplanation()
        {
            GUILayout.Label(
                PlasticLocalization.Name.LeaveChangesExplanation.GetString(),
                UnityStyles.Dialog.MessageText);
        }

        void DrawBringRadioToggleExplanation()
        {
            GUILayout.Label(
                PlasticLocalization.Name.BringChangesExplanation.GetString(),
                UnityStyles.Dialog.MessageText);
        }

        void DoButtonsArea()
        {
            using (new EditorGUILayout.HorizontalScope())
            {
                GUILayout.FlexibleSpace();

                if (Application.platform == RuntimePlatform.WindowsEditor)
                {
                    DoSwitchButton();
                    DoCancelButton();
                    return;
                }

                DoCancelButton();
                DoSwitchButton();
            }
        }

        void DoSwitchButton()
        {
            if (!NormalButton(PlasticLocalization.Name.SwitchButton.GetString()))
                return;

            DoSwitchAction();
        }

        void DoCancelButton()
        {
            if (!NormalButton(PlasticLocalization.Name.CancelButton.GetString()))
                return;

            CancelButtonAction();
        }

        void DoSwitchAction()
        {
            if (mWarnOverwriteExistingShelves &&
                mLeaveRadioToggle &&
                !AskForOverwriteConfirmation())
                return;

            if (mLeaveRadioToggle)
            {
                CloseButtonAction();
                return;
            }

            OkButtonAction();
        }

        static void DrawTabbedContent(Action drawContent)
        {
            float originalLabelWidth = EditorGUIUtility.labelWidth;

            try
            {
                using (new EditorGUILayout.HorizontalScope())
                {
                    EditorGUILayout.Space(21);
                    EditorGUIUtility.labelWidth -= 21;
                    using (new EditorGUILayout.VerticalScope())
                    {
                        GUILayout.Space(0);
                        drawContent();
                    }

                    GUILayout.FlexibleSpace();
                }
            }
            finally
            {
                EditorGUIUtility.labelWidth = originalLabelWidth;
            }
        }

        bool mLeaveRadioToggle = true;
        bool mBringRadioToggle;

        string mSrcObject;
        string mDstObject;
        bool mCanBringPendingChanges;
        bool mWarnOverwriteExistingShelves;
    }
}
