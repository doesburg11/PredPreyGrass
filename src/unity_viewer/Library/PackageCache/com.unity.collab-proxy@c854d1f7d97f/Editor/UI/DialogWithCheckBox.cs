using UnityEditor;
using UnityEngine;

using Codice.Client.Common;

namespace Unity.PlasticSCM.Editor.UI
{
    internal class DialogWithCheckBox : PlasticDialog
    {
        protected override Rect DefaultRect
        {
            get
            {
                var baseRect = base.DefaultRect;
                return new Rect(baseRect.x, baseRect.y, 535, baseRect.height);
            }
        }

        internal static GuiMessage.GuiMessageResponseButton Show(
            string title,
            string message,
            string positiveButtonText,
            string neutralButtonText,
            string negativeButtonText,
            MultiLinkLabelData dontShowAgainContent,
            EditorWindow parentWindow,
            out bool checkBoxValue)
        {
            checkBoxValue = false;

            DialogWithCheckBox dialog = Create(
                title,
                message,
                positiveButtonText,
                neutralButtonText,
                negativeButtonText,
                dontShowAgainContent);

            ResponseType result = dialog.RunModal(parentWindow);
            if (result == ResponseType.Cancel || result == ResponseType.None)
                return GuiMessage.GuiMessageResponseButton.Neutral;

            checkBoxValue = dialog.mCheckBox;
            if (result == ResponseType.Ok)
                return GuiMessage.GuiMessageResponseButton.Positive;

            return GuiMessage.GuiMessageResponseButton.Negative;
        }

        protected override string GetTitle()
        {
            return mTitle;
        }

        protected override void OnModalGUI()
        {
            DoMainContentSection();

            DoButtonsArea();

            DoCheckboxSection();
        }

        void DoMainContentSection()
        {
            using (new EditorGUILayout.VerticalScope())
            {
                GUILayout.Label(
                    mTitle,
                    UnityStyles.Dialog.MessageTitle);

                GUILayout.Space(3f);

                GUILayout.Label(
                    mMessage,
                    UnityStyles.Dialog.MessageText);
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
                        DoPositiveButton();
                        DoNegativeButton();
                        DoNeutralButton();
                        return;
                    }

                    DoNegativeButton();
                    DoNeutralButton();
                    DoPositiveButton();
                }
            }
        }

        void DoPositiveButton()
        {
            GUILayout.Space(6f);

            if (!AcceptButton(mPositiveButtonText,
                30))
                return;

            OkButtonAction();
        }

        void DoNegativeButton()
        {
            if (string.IsNullOrEmpty(mNegativeButtonText))
                return;

            GUILayout.Space(6f);

            if (!NormalButton(mNegativeButtonText))
                return;

            ApplyButtonAction();
        }

        void DoNeutralButton()
        {
            if (string.IsNullOrEmpty(mNeutralButtonText))
                return;

            GUILayout.Space(6f);

            if (!NormalButton(mNeutralButtonText))
                return;

            CancelButtonAction();
        }

        void DoCheckboxSection()
        {
            GUILayout.Space(22f);

            Rect backgroundRect = new Rect(0, GUILayoutUtility.GetLastRect().yMax, position.width, 50);

            EditorGUI.DrawRect(backgroundRect, UnityStyles.Colors.DarkGray);

            GUILayout.Space(4f);

            using (new EditorGUILayout.HorizontalScope())
            {
                mCheckBox = EditorGUILayout.ToggleLeft(
                    string.Empty,
                    mCheckBox,
                    EditorStyles.boldLabel);
                GUILayout.FlexibleSpace();
            }

            GUILayout.Space(-22);

            using (new EditorGUILayout.HorizontalScope())
            {
                GUILayout.Space(22);
                DrawTextBlockWithLink.ForMultiLinkLabelInDialog(mDontShowAgainContent);
            }

            GUILayout.Space(-19);
        }

        static DialogWithCheckBox Create(
            string title,
            string message,
            string positiveButtonText,
            string neutralButtonText,
            string negativeButtonText,
            MultiLinkLabelData dontShowAgainContent)
        {
            DialogWithCheckBox instance = CreateInstance<DialogWithCheckBox>();
            instance.mEnterKeyAction = instance.OkButtonAction;
            instance.mEscapeKeyAction = instance.CancelButtonAction;

            instance.mTitle = title;
            instance.mMessage = message;
            instance.mPositiveButtonText = positiveButtonText;
            instance.mNeutralButtonText = neutralButtonText;
            instance.mNegativeButtonText = negativeButtonText;
            instance.mDontShowAgainContent = dontShowAgainContent;

            return instance;
        }

        string mTitle;
        string mMessage;
        string mPositiveButtonText;
        string mNeutralButtonText;
        string mNegativeButtonText;
        MultiLinkLabelData mDontShowAgainContent;

        bool mCheckBox;
    }
}
