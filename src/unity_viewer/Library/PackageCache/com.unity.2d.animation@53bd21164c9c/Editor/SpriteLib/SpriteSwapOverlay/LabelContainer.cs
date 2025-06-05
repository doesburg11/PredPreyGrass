using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.U2D.Animation.SceneOverlays
{
    internal class LabelContainer : VisualElement, INavigableElement
    {
        static class Styles
        {
            public const string labelVisual = SpriteSwapOverlay.rootStyle + "__label-visual";
            public const string labelContainer = SpriteSwapOverlay.rootStyle + "__label-container";
            public const string directionButton = SpriteSwapOverlay.rootStyle + "__label-direction-button";
            public const string labelImagesContainer = SpriteSwapOverlay.rootStyle + "__label-images-container";
            public const string labelSelected = SpriteSwapOverlay.rootStyle + "__label-selected";
        }

        const string k_ScrollLeftIcon = "scrollleft_uielements";
        const string k_ScrollRightIcon = "scrollright_uielements";
        const string k_ScrollLeftIconDark = "d_scrollleft_uielements";
        const string k_ScrollRightIconDark = "d_scrollright_uielements";

        public event Action<int> onSelectionChange;

        public int itemCount => m_LabelImagesContainer.childCount;
        public int selectedIndex { get; private set; } = -1;

        public VisualElement visualElement => this;

        List<Tuple<string, Sprite>> m_Labels;

        Button m_PreviousButton;
        Button m_NextButton;
        ScrollView m_LabelImagesContainer;

        bool m_IsFocused;

        PropertyAnimationState m_AnimationState;

        public LabelContainer()
        {
            m_Labels = new List<Tuple<string, Sprite>>();

            Texture2D previousButtonIcon;
            Texture2D nextButtonIcon;
            if (EditorGUIUtility.isProSkin)
            {
                previousButtonIcon = (Texture2D)EditorGUIUtility.IconContent(k_ScrollLeftIconDark).image;
                nextButtonIcon = (Texture2D)EditorGUIUtility.IconContent(k_ScrollRightIconDark).image;
            }
            else
            {
                previousButtonIcon = (Texture2D)EditorGUIUtility.IconContent(k_ScrollLeftIcon).image;
                nextButtonIcon = (Texture2D)EditorGUIUtility.IconContent(k_ScrollRightIcon).image;
            }

            m_LabelImagesContainer = new ScrollView
            {
                horizontalScrollerVisibility = ScrollerVisibility.Hidden,
                nestedInteractionKind = ScrollView.NestedInteractionKind.StopScrolling,
                mode = ScrollViewMode.Horizontal
            };
            m_LabelImagesContainer.AddToClassList(Styles.labelImagesContainer);

            m_PreviousButton = new Button { style = { backgroundImage = Background.FromTexture2D(previousButtonIcon) } };
            m_PreviousButton.clicked += () => m_LabelImagesContainer.horizontalScroller.ScrollPageUp();
            m_PreviousButton.AddToClassList(Styles.directionButton);
            m_PreviousButton.RemoveFromClassList(Button.ussClassName);

            m_NextButton = new Button { style = { backgroundImage = Background.FromTexture2D(nextButtonIcon) } };
            m_NextButton.clicked += () => m_LabelImagesContainer.horizontalScroller.ScrollPageDown();
            m_NextButton.AddToClassList(Styles.directionButton);
            m_NextButton.RemoveFromClassList(Button.ussClassName);

            Add(m_PreviousButton);
            Add(m_LabelImagesContainer);
            Add(m_NextButton);

            AddToClassList(Styles.labelContainer);
            RegisterCallback<GeometryChangedEvent>(OnGeometryChanged);
        }

        public void SetItems(IList labels)
        {
            ClearItems();

            if (labels == null)
                return;

            m_Labels = labels as List<Tuple<string, Sprite>>;
            if (m_Labels == null)
                return;

            foreach (var (labelName, labelSprite) in m_Labels)
                m_LabelImagesContainer.Add(GetVisualForLabel(labelName, labelSprite));
        }

        public object GetItem(int index)
        {
            if (m_Labels == null || index < 0 || index >= m_Labels.Count)
                return null;

            return m_Labels[index];
        }

        public void Select(int index)
        {
            if (index < 0 || index >= itemCount)
                return;

            selectedIndex = index;
            UpdateSelectionVisuals();
            onSelectionChange?.Invoke(selectedIndex);
        }

        void OnGeometryChanged(GeometryChangedEvent evt)
        {
            UpdateElementSize();
            UpdateSelectionVisuals();
            UpdateNavigationButtons();
        }

        void UpdateElementSize()
        {
            var paddingAndMarginsSize = 4.0f;
            style.minHeight = style.maxHeight = SpriteSwapOverlay.Settings.thumbnailSize + paddingAndMarginsSize;
        }

        void ClearItems()
        {
            m_Labels.Clear();
            selectedIndex = -1;
            m_LabelImagesContainer.Clear();
        }

        void OnLabelPointerDown(PointerDownEvent evt)
        {
            if (evt.currentTarget is Image image)
                Select(m_LabelImagesContainer.IndexOf(image));
        }

        void UpdateSelectionVisuals()
        {
            foreach (var child in m_LabelImagesContainer.Children())
            {
                if (m_LabelImagesContainer.IndexOf(child) == selectedIndex)
                {
                    m_LabelImagesContainer.ScrollTo(child);
                    child.AddToClassList(Styles.labelSelected);

                    child.style.backgroundColor = GetColorForAnimationState(m_AnimationState);
                }
                else
                {
                    child.RemoveFromClassList(Styles.labelSelected);
                }
            }
        }

        void UpdateNavigationButtons()
        {
            var enableNavigationButtons = m_LabelImagesContainer.contentContainer.contentRect.width > m_LabelImagesContainer.contentRect.width;
            m_PreviousButton.SetEnabled(enableNavigationButtons);
            m_NextButton.SetEnabled(enableNavigationButtons);
        }

        VisualElement GetVisualForLabel(string labelName, Sprite labelSprite)
        {
            var image = new Image { name = labelName, tooltip = labelName, sprite = labelSprite };
            image.style.height = image.style.width = SpriteSwapOverlay.Settings.thumbnailSize;
            image.AddToClassList(Styles.labelVisual);
            image.RegisterCallback<PointerDownEvent>(OnLabelPointerDown);
            return image;
        }

        public void SetAnimationState(PropertyAnimationState animationState)
        {
            if (animationState != m_AnimationState)
            {
                m_AnimationState = animationState;

                UpdateSelectionVisuals();
            }
        }

        static Color GetColorForAnimationState(PropertyAnimationState animationState)
        {
            var color = new Color(88.0f / 255.0f, 88.0f / 255.0f, 88.0f / 255.5f, 1.0f);
            if (animationState == PropertyAnimationState.Animated)
                color *= AnimationMode.animatedPropertyColor;
            else if (animationState == PropertyAnimationState.Candidate)
                color *= AnimationMode.candidatePropertyColor;
            else if (animationState == PropertyAnimationState.Recording)
                color *= AnimationMode.recordedPropertyColor;

            return color;
        }
    }
}
