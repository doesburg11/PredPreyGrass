using System;
using System.Collections.Generic;

namespace UnityEditor.Tilemaps
{
    /// <summary>
    /// An attribute for GridBrushBase which specifies the TilemapEditorTool types which can work with the GridBrushBase.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class)]
    public class BrushToolsAttribute : Attribute
    {
        private List<Type> m_ToolTypes;
        internal List<Type> toolList
        {
            get { return m_ToolTypes; }
        }

        /// <summary>
        /// Constructor for BrushToolsAttribute. Specify the TilemapEditorTool types which can work with the GridBrushBase.
        /// </summary>
        /// <param name="tools">An array of TilemapEditorTool types which can work with the GridBrushBase.</param>
        public BrushToolsAttribute(params Type[] tools)
        {
            m_ToolTypes = new List<Type>();
            foreach (var toolType in tools)
            {
                if (toolType.IsSubclassOf(typeof(TilemapEditorTool)) && !m_ToolTypes.Contains(toolType))
                {
                    m_ToolTypes.Add(toolType);
                }
            }
        }
    }
}
