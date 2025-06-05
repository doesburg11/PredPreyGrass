// This class is taken from UnityEngine.UIElements.ButtonStripField

using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;
using UnityEngine.U2D.Common;

namespace UnityEditor.U2D.Animation.Upgrading
{
#if ENABLE_UXML_SERIALIZED_DATA
    [UxmlElement]
#endif
    internal partial class ButtonStripField : BaseField<int>
    {
#if ENABLE_UXML_TRAITS
        public new class UxmlFactory : UxmlFactory<ButtonStripField, UxmlTraits> {}
        public new class UxmlTraits : BaseField<int>.UxmlTraits {}
#endif

        const string k_ClassName = "unity-button-strip-field";
        const string k_ButtonClass = k_ClassName + "__button";
        const string k_IconClass = k_ClassName + "__button-icon";
        const string k_ButtonLeftClass = k_ButtonClass + "--left";
        const string k_ButtonMiddleClass = k_ButtonClass + "--middle";
        const string k_ButtonRightClass = k_ButtonClass + "--right";
        const string k_ButtonAloneClass = k_ButtonClass + "--alone";

        List<Button> m_Buttons = new List<Button>();

        /// <summary>
        /// Add a button to the button strip.
        /// </summary>
        /// <param name="text">Text to be displayed.</param>
        /// <param name="name">Name of the element.</param>
        public void AddButton(string text, string name = "")
        {
            var button = CreateButton(name);
            button.text = text;
            Add(button);
        }

        /// <summary>
        /// Add a button to the button strip.
        /// </summary>
        /// <param name="icon">Icon used for the button.</param>
        /// <param name="name">Name of the element.</param>
        public void AddButton(Background icon, string name = "")
        {
            var button = CreateButton(name);
            var iconElement = new VisualElement();
            iconElement.AddToClassList(k_IconClass);
            iconElement.style.backgroundImage = icon;
            button.Add(iconElement);
            Add(button);
        }

        Button CreateButton(string name)
        {
            var button = new Button { name = name, };

            button.AddToClassList(k_ButtonClass);
            button.RegisterCallback<DetachFromPanelEvent>(OnButtonDetachFromPanel);
            button.clicked += () => { value = m_Buttons.IndexOf(button); };

            m_Buttons.Add(button);
            Add(button);

            RefreshButtonsStyling();
            return button;
        }

        static void OnButtonDetachFromPanel(DetachFromPanelEvent evt)
        {
            if (evt.currentTarget is VisualElement element
                && element.parent is ButtonStripField buttonStrip)
            {
                buttonStrip.RefreshButtonsStyling();
                buttonStrip.EnsureValueIsValid();
            }
        }

        void RefreshButtonsStyling()
        {
            for (var i = 0; i < m_Buttons.Count; ++i)
            {
                var button = m_Buttons[i];
                bool alone = m_Buttons.Count == 1;
                bool left = i == 0;
                bool right = i == m_Buttons.Count - 1;

                button.EnableInClassList(k_ButtonAloneClass, alone);
                button.EnableInClassList(k_ButtonLeftClass, !alone && left);
                button.EnableInClassList(k_ButtonRightClass, !alone && right);
                button.EnableInClassList(k_ButtonMiddleClass, !alone && !left && !right);
            }
        }

        /// <summary>
        /// Constructs a <see cref="ButtonStripField"/>, with all required properties provided.
        /// </summary>
        public ButtonStripField() : base("", null)
        {
        }

        /// <summary>
        /// Constructs a <see cref="ButtonStripField"/>, with all required properties provided.
        /// </summary>
        /// <param name="label">The list of items to use as a data source.</param>
        public ButtonStripField(string label) : base(label, null)
        {
            AddToClassList(k_ClassName);
        }

        public override void SetValueWithoutNotify(int newValue)
        {
            newValue = Mathf.Clamp(newValue, 0, m_Buttons.Count - 1);
            base.SetValueWithoutNotify(newValue);
            RefreshButtonsState();
        }

        void EnsureValueIsValid()
        {
            SetValueWithoutNotify(Mathf.Clamp(value, 0, m_Buttons.Count - 1));
        }

        void RefreshButtonsState()
        {
            for (var i = 0; i < m_Buttons.Count; ++i)
                m_Buttons[i].SetChecked(i == value);
        }
    }
}