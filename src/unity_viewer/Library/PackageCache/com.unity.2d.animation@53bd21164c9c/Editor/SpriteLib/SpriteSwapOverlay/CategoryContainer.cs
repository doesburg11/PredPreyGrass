using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.U2D.Animation.SceneOverlays
{
    internal class CategoryContainer : DropdownField, INavigableElement
    {
        static class Styles
        {
            public const string categoryContainer = SpriteSwapOverlay.rootStyle + "__category-container";
        }

        public event Action<int> onSelectionChange;

        public int itemCount => choices.Count;
        public int selectedIndex { get; private set; } = -1;

        public VisualElement visualElement => this;

        public CategoryContainer()
        {
            tooltip = TextContent.spriteLibraryCategoriesTooltip;
            label = string.Empty;
            this.RegisterValueChangedCallback(OnValueChanged);
            AddToClassList(Styles.categoryContainer);
        }

        public void SetItems(IList categories)
        {
            var newChoices = (List<string>)categories ?? new List<string>();
            newChoices.Insert(0, TextContent.noCategory);
            choices = newChoices;
            SetEnabled(newChoices.Count > 1);
        }

        public void Select(int i)
        {
            if (choices == null || i >= choices.Count || i < 0)
                value = TextContent.noCategory;
            else
                value = choices[i];

            tooltip = value;
        }

        public object GetItem(int i)
        {
            if (choices == null || i >= choices.Count || i <= 0)
                return string.Empty;

            return choices[i];
        }

        void OnValueChanged(ChangeEvent<string> evt)
        {
            var newSelection = evt.newValue;
            if (string.IsNullOrWhiteSpace(newSelection))
                return;

            var newIndex = choices.IndexOf(newSelection);
            if (newIndex == selectedIndex)
                return;

            selectedIndex = newIndex;

            onSelectionChange?.Invoke(selectedIndex);
        }
    }
}
