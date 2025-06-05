using System;
using System.Collections.Generic;
using UnityEditor.U2D.Common;
using UnityEditor.UIElements;
using UnityEngine;
using UnityEngine.U2D.Animation;
using UnityEngine.UIElements;

namespace UnityEditor.U2D.Animation.SceneOverlays
{
    class SpriteResolverSelector : VisualElement
    {
        static class Styles
        {
            public const string spriteResolverNameLabel = SpriteSwapOverlay.rootStyle + "__resolver-name-label";
            public const string categoryAndLabelNameContainer = SpriteSwapOverlay.rootStyle + "__category-and-label-name-container";
            public const string selector = SpriteSwapOverlay.rootStyle + "__selector";
            public const string descriptionLabel = SpriteSwapOverlay.rootStyle + "__label-description";
        }

        Label m_SpriteResolverLabel;

        INavigableElement m_CategoryContainer;
        INavigableElement m_LabelContainer;
        INavigableElement m_CurrentSelection;

        Label m_LabelNameLabel;

        string m_Category = string.Empty;
        string m_Label = string.Empty;

        List<string> m_AvailableCategories;
        List<Tuple<string, Sprite>> m_AvailableLabels;

        SerializedObject m_SerializedResolver;
        SerializedProperty m_SpriteHashProperty;
        SpriteResolver spriteResolver => (SpriteResolver)m_SerializedResolver?.targetObject;

        public SpriteResolverSelector(INavigableElement categoryContainer, INavigableElement labelContainer)
        {
            AddToClassList(Styles.selector);

            m_SpriteResolverLabel = new Label();
            m_SpriteResolverLabel.AddToClassList(Styles.spriteResolverNameLabel);
            Add(m_SpriteResolverLabel);

            var categoryAndLabelNameHolder = new VisualElement();
            categoryAndLabelNameHolder.AddToClassList(Styles.categoryAndLabelNameContainer);
            Add(categoryAndLabelNameHolder);

            m_CategoryContainer = categoryContainer;
            m_CategoryContainer.onSelectionChange += OnCategorySelected;
            var categoryContainerVisual = m_CategoryContainer.visualElement;
            categoryContainerVisual.RegisterCallback<FocusInEvent>(OnFocusIn);
            categoryContainerVisual.RegisterCallback<FocusOutEvent>(OnFocusOut);
            categoryAndLabelNameHolder.Add(categoryContainerVisual);

            m_LabelNameLabel = new Label();
            m_LabelNameLabel.AddToClassList(Styles.descriptionLabel);
            categoryAndLabelNameHolder.Add(m_LabelNameLabel);

            m_LabelContainer = labelContainer;
            m_LabelContainer.onSelectionChange += OnLabelSelected;
            var labelContainerVisual = m_LabelContainer.visualElement;
            labelContainerVisual.RegisterCallback<FocusInEvent>(OnFocusIn);
            labelContainerVisual.RegisterCallback<FocusOutEvent>(OnFocusOut);
            Add(labelContainerVisual);

            RegisterCallback<KeyDownEvent>(OnKeyDown);
        }

        public void Select()
        {
            m_LabelContainer.visualElement.Focus();
        }

        public void SceneUpdate()
        {
            UpdateAnimationColor();
        }

        public void SetSpriteResolver(SpriteResolver newSpriteResolver)
        {
            this.Unbind();

            if (newSpriteResolver != null)
            {
                m_SerializedResolver = new SerializedObject(newSpriteResolver);
                m_SpriteHashProperty = m_SerializedResolver.FindProperty(SpriteResolver.spriteHashPropertyName);
                this.TrackPropertyValue(m_SpriteHashProperty, OnResolvedSprite);
                ReadCategoryAndLabelFromSelection();
            }
            else
            {
                m_SerializedResolver = null;
                m_SpriteHashProperty = null;
            }
        }

        void OnResolvedSprite(SerializedProperty serializedProperty)
        {
            ReadCategoryAndLabelFromSelection();
        }

        void ReadCategoryAndLabelFromSelection()
        {
            var resolver = spriteResolver;
            if (resolver == null)
                return;

            m_Category = resolver.GetCategory() ?? string.Empty;
            m_Label = resolver.GetLabel() ?? string.Empty;

            UpdateVisuals();
        }

        void UpdateVisuals()
        {
            var resolver = spriteResolver;
            if (resolver == null)
                return;

            m_SpriteResolverLabel.text = m_SpriteResolverLabel.tooltip = resolver.name;

            m_AvailableCategories = GetAvailableCategories(resolver) ?? new List<string>();
            m_AvailableLabels = new List<Tuple<string, Sprite>>();
            if (resolver.spriteLibrary != null)
            {
                foreach (var labelName in GetAvailableLabels(resolver, m_Category))
                    m_AvailableLabels.Add(new Tuple<string, Sprite>(labelName, resolver.spriteLibrary.GetSprite(m_Category, labelName)));
            }

            m_CategoryContainer.SetItems(m_AvailableCategories);
            m_CategoryContainer.Select(m_AvailableCategories.IndexOf(m_Category));

            m_LabelContainer.SetItems(m_AvailableLabels);
            m_LabelContainer.Select(m_AvailableLabels.FindIndex(label => label.Item1 == m_Label));

            m_Label = !string.IsNullOrWhiteSpace(m_Label) ? m_Label : TextContent.emptyCategory;
            m_LabelNameLabel.text = m_LabelNameLabel.tooltip = m_Label;
            m_LabelNameLabel.SetEnabled(m_AvailableLabels.Count > 0);

            if (m_LabelContainer.itemCount == 0)
                m_CurrentSelection = m_CategoryContainer;

            UpdateAnimationColor();
        }

        void OnCategorySelected(int newSelection)
        {
            var resolver = spriteResolver;
            if (resolver == null)
                return;

            var categoryName = (string)m_CategoryContainer.GetItem(newSelection);
            if (categoryName == null || categoryName == m_Category)
                return;

            var availableLabels = resolver.spriteLibrary != null ? resolver.spriteLibrary.GetEntryNames(categoryName) : null;
            var labelList = availableLabels != null ? new List<string>(availableLabels) : new List<string>();
            var labelName = string.Empty;
            if (labelList.Count > 0)
                labelName = labelList.Contains(m_Label) ? m_Label : labelList[0];

            m_SpriteHashProperty.intValue = SpriteLibrary.GetHashForCategoryAndEntry(categoryName, labelName);
            m_SerializedResolver.ApplyModifiedProperties();

            ReadCategoryAndLabelFromSelection();
        }

        void OnLabelSelected(int newSelection)
        {
            var resolver = spriteResolver;
            if (resolver == null)
                return;

            var (labelName, _) = (Tuple<string, Sprite>)m_LabelContainer.GetItem(newSelection);
            if (string.IsNullOrWhiteSpace(labelName) || labelName == m_Label)
                return;

            m_SpriteHashProperty.intValue = SpriteLibrary.GetHashForCategoryAndEntry(m_Category, labelName);
            m_SerializedResolver.ApplyModifiedProperties();
            m_LabelNameLabel.text = m_LabelNameLabel.tooltip = labelName;

            ReadCategoryAndLabelFromSelection();
        }

        void OnKeyDown(KeyDownEvent evt)
        {
            if (m_CurrentSelection == null)
                return;

            switch (evt.keyCode)
            {
                case KeyCode.LeftArrow:
                    var previousIndex = m_CurrentSelection.selectedIndex - 1;
                    if (previousIndex < 0)
                        previousIndex += m_CurrentSelection.itemCount;
                    m_CurrentSelection.Select(previousIndex);
                    evt.StopPropagation();
                    break;
                case KeyCode.RightArrow:
                    var nextIndex = m_CurrentSelection.selectedIndex + 1;
                    if (nextIndex >= m_CurrentSelection.itemCount)
                        nextIndex = 0;
                    m_CurrentSelection?.Select(nextIndex);
                    evt.StopPropagation();
                    break;
                case KeyCode.DownArrow:
                case KeyCode.UpArrow:
                    evt.StopPropagation();
                    break;
            }
        }

        void OnFocusIn(FocusInEvent evt)
        {
            var navigable = (INavigableElement)evt.currentTarget;
            if (navigable != null)
                m_CurrentSelection = navigable;
        }

        void OnFocusOut(FocusOutEvent evt)
        {
            var navigable = (INavigableElement)evt.currentTarget;
            if (navigable != null && m_CurrentSelection == navigable)
                m_CurrentSelection = null;
        }

        static List<string> GetAvailableCategories(SpriteResolver spriteResolver)
        {
            if (spriteResolver == null || spriteResolver.spriteLibrary == null)
                return new List<string>();

            var availableCategories = spriteResolver.spriteLibrary.categoryNames;
            return availableCategories != null ? new List<string>(availableCategories) : new List<string>();
        }

        static List<string> GetAvailableLabels(SpriteResolver spriteResolver, string categoryName)
        {
            if (spriteResolver == null || spriteResolver.spriteLibrary == null || string.IsNullOrEmpty(categoryName))
                return new List<string>();

            var availableLabels = spriteResolver.spriteLibrary.GetEntryNames(categoryName);
            return availableLabels != null ? new List<string>(availableLabels) : new List<string>();
        }

        void UpdateAnimationColor()
        {
            var spriteResolverObject = spriteResolver;
            var animationState = PropertyAnimationState.NotAnimated;
            if (spriteResolverObject != null && m_SpriteHashProperty != null)
            {
                if (InternalEditorBridge.IsAnimated(spriteResolverObject, m_SpriteHashProperty))
                {
                    if (InternalEditorBridge.InAnimationRecording())
                        animationState = PropertyAnimationState.Recording;
                    else if (InternalEditorBridge.IsCandidate(spriteResolverObject, m_SpriteHashProperty))
                        animationState = PropertyAnimationState.Candidate;
                    else
                        animationState = PropertyAnimationState.Animated;
                }
            }

            (m_LabelContainer as LabelContainer)?.SetAnimationState(animationState);
        }
    }
}
