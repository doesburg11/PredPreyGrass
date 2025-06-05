using System;
using UnityEditor;
using UnityEditor.UIElements;
using UnityEngine.UIElements;

namespace Unity.PerformanceTesting.Editor.UIElements
{
    internal class ToolbarWithSearch : VisualElement
    {
        private string m_searchString;
        public Action<string> SearchTextChanged;

        public ToolbarWithSearch()
        {
            // Create the toolbar and search field elements
            var toolbar = new Toolbar();
            var searchField = new ToolbarSearchField();

            // Add the toolbar and search field elements to this element
            Add(toolbar);
            Add(searchField);
        }

        public void Draw()
        {
            EditorGUILayout.BeginHorizontal(EditorStyles.toolbar);
            EditorGUI.BeginChangeCheck();
            m_searchString = EditorGUILayout.TextField(m_searchString, EditorStyles.toolbarSearchField);
            if (EditorGUI.EndChangeCheck()) SearchTextChanged?.Invoke(m_searchString);
            EditorGUILayout.EndHorizontal();
        }

        public void ClearSearchString()
        {
            m_searchString = string.Empty;
            SearchTextChanged?.Invoke(m_searchString);
        }
    }
}
