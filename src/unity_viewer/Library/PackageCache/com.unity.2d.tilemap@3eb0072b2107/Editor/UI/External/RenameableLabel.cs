using System;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps.External
{
    /// <summary>
    /// Label that can be renamed.
    /// </summary>
    class RenameableLabel : VisualElement
    {
        static readonly string k_USSFilePath = "Packages/com.unity.2d.tilemap/Editor/UI/External/RenameableLabel.uss";

        public static readonly string ussClassName = "u2d-renameable-label";

        /// <summary>
        /// Text to display.
        /// </summary>
        internal string text
        {
            get => label.text;
            set => label.text = value;
        }

        internal bool isRenaming { get; set; }

        internal event Action<RenameableLabel, bool> renameEnding;

        Label label { get; } = new();
        TextField textField { get; } = new();

        internal RenameableLabel() : base()
        {
            name = ussClassName;
            AddToClassList(ussClassName);

            styleSheets.Add(AssetDatabase.LoadAssetAtPath<StyleSheet>(k_USSFilePath));

            focusable = true;
            delegatesFocus = false;

            Add(label);
            Add(textField);

            label.RegisterCallback<MouseUpEvent>(OnLabelMouseUpEvent);

            textField.selectAllOnFocus = true;
            textField.selectAllOnMouseUp = false;
            textField.style.display = DisplayStyle.None;

            textField.RegisterCallback<MouseUpEvent>(OnMouseUpEvent);
            textField.RegisterCallback<KeyDownEvent>(OnKeyDownEvent);
            textField.RegisterCallback<BlurEvent>(OnBlurEvent);
        }

        internal void BeginRename()
        {
            if (isRenaming)
                return;

            isRenaming = true;
            delegatesFocus = true;

            label.style.display = DisplayStyle.None;
            textField.style.display = DisplayStyle.Flex;

            textField.value = text;
            textField.Q<TextElement>().Focus();
        }

        internal void CancelRename()
        {
            if (isRenaming)
                EndRename(false);
        }

        void EndRename(bool canceled = false)
        {
            isRenaming = false;
            delegatesFocus = false;
            schedule.Execute(Focus);

            textField.style.display = DisplayStyle.None;
            label.style.display = DisplayStyle.Flex;

            if (!canceled) // When the rename is canceled, the label keep its current value.
                label.text = textField.value;

            renameEnding?.Invoke(this, canceled);
        }

        void OnLabelMouseUpEvent(MouseUpEvent evt)
        {
            if (isRenaming)
                return;

            BeginRename();
            evt.StopPropagation();
        }

        void OnMouseUpEvent(MouseUpEvent evt)
        {
            if (!isRenaming)
                return;

            textField.Q<TextElement>().Focus();
            evt.StopPropagation();
        }

        void OnKeyDownEvent(KeyDownEvent evt)
        {
            if (isRenaming && evt.keyCode == KeyCode.Escape)
                EndRename(true);
        }

        void OnBlurEvent(BlurEvent evt)
        {
            if (!isRenaming)
                return;

            EndRename();
        }
    }
}
