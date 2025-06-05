using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.U2D.Animation;
using UnityEngine.UIElements;

namespace UnityEditor.U2D.Animation.SceneOverlays
{
    internal class SpriteSwapVisualElement : VisualElement
    {
        static class Styles
        {
            public const string selectorList = SpriteSwapOverlay.rootStyle + "__selector-list";
            public static string infoLabelHolder = SpriteSwapOverlay.rootStyle + "__info-label-holder";
            public static string infoLabelIcon = SpriteSwapOverlay.rootStyle + "__info-label-icon";
            public static string infoLabel = SpriteSwapOverlay.rootStyle + "__info-label";
        }

        VisualElement m_InfoLabelHolder;
        Image m_InfoIcon;
        Label m_InfoLabel;
        ListView m_ListView;
        SpriteResolver[] m_Selection = Array.Empty<SpriteResolver>();

        public SpriteSwapVisualElement()
        {
            AddToClassList(SpriteSwapOverlay.rootStyle);

            m_InfoLabelHolder = new VisualElement();
            m_InfoLabelHolder.AddToClassList(Styles.infoLabelHolder);
            m_InfoIcon = new Image { image = EditorGUIUtility.IconContent("console.infoicon").image };
            m_InfoIcon.AddToClassList(Styles.infoLabelIcon);
            m_InfoLabel = new Label { text = TextContent.spriteSwapSelectSpriteResolver };
            m_InfoLabel.AddToClassList(Styles.infoLabel);
            m_InfoLabelHolder.Add(m_InfoIcon);
            m_InfoLabelHolder.Add(m_InfoLabel);
            Add(m_InfoLabelHolder);

            m_ListView = new ListView { virtualizationMethod = CollectionVirtualizationMethod.DynamicHeight };
            m_ListView.makeItem += MakeItem;
            m_ListView.bindItem += BindItem;
            m_ListView.unbindItem += UnbindItem;
            m_ListView.selectionChanged += OnSelectionChanged;
            m_ListView.AddToClassList(Styles.selectorList);
            Add(m_ListView);

            Add(new OverlayToolbar());
        }

        static VisualElement MakeItem() =>
            new SpriteResolverSelector
                (
                    new CategoryContainer(),
                    new LabelContainer() { focusable = true }
                )
                { focusable = true };

        void BindItem(VisualElement visualElement, int i)
        {
            if (m_ListView.itemsSource == null || m_ListView.itemsSource.Count <= i)
                return;

            var resolverSelector = (SpriteResolverSelector)visualElement;
            resolverSelector.SetSpriteResolver((SpriteResolver)m_ListView.itemsSource[i]);
        }

        static void UnbindItem(VisualElement visualElement, int i)
        {
            var resolverSelector = (SpriteResolverSelector)visualElement;
            resolverSelector.SetSpriteResolver(null);
        }

        void OnSelectionChanged(IEnumerable<object> obj)
        {
            var index = m_ListView.selectedIndex;
            if (index == -1)
                return;

            m_ListView.SetSelectionWithoutNotify(new[] { index });

            var selector = (SpriteResolverSelector)m_ListView.GetRootElementForIndex(index);
            selector?.Select();
        }

        public void SetSpriteResolvers(SpriteResolver[] newSelection)
        {
            var isListVisible = newSelection is { Length: > 0 };

            m_InfoLabelHolder.style.display = isListVisible ? DisplayStyle.None : DisplayStyle.Flex;
            m_ListView.style.display = isListVisible ? DisplayStyle.Flex : DisplayStyle.None;

            if (!AreCollectionsEqual(m_Selection, newSelection))
            {
                m_ListView.selectedIndex = -1;
                m_ListView.itemsSource = newSelection;
                m_ListView.Rebuild();
            }

            m_Selection = newSelection;
        }

        public void RefreshSpriteResolvers()
        {
            m_ListView.Rebuild();
        }

        public void OnSceneGUI()
        {
            if (m_ListView?.itemsSource == null)
                return;

            for (var i = 0; i < m_ListView.itemsSource.Count; i++)
            {
                var id = m_ListView.viewController.GetIdForIndex(i);
                var spriteResolverSelector = (SpriteResolverSelector)m_ListView.GetRootElementForId(id);
                spriteResolverSelector?.SceneUpdate();
            }
        }

        public void SetFiltered(bool isFiltered)
        {
            m_InfoLabel.text = isFiltered ? TextContent.spriteSwapFilteredContent : TextContent.spriteSwapSelectSpriteResolver;
            m_InfoIcon.style.display = isFiltered ? DisplayStyle.Flex : DisplayStyle.None;
        }

        static bool AreCollectionsEqual(IList first, IList second)
        {
            if (first == null || second == null)
                return false;

            if (first.Count != second.Count)
                return false;

            for (var i = 0; i < first.Count; i++)
            {
                if (!first[i].Equals(second[i]))
                    return false;
            }

            return true;
        }
    }
}
