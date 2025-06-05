using UnityEngine;
using UnityEngine.UIElements;
using UnityEditor.U2D.Common;

namespace UnityEditor.U2D.PSD
{
    internal class UICellLabelElement : UICellElement
     {
         Label m_Label;
         VisualElement m_FolderIcon;
         public UICellLabelElement()
         {
             m_FolderIcon = new VisualElement()
             {
                 name = "UICellFolderElement"
             };
             m_Label = new Label()
             {
                 name = "UICellLabelElement"
             };
             this.Add(m_FolderIcon);
             this.Add(m_Label);
         }

         public string text
         {
             set { m_Label.text = value; }
         }

         public void EnableFolderIcon(Texture2D v)
         {
             if (v != null)
             {
                 m_FolderIcon.SetHiddenFromLayout(false);
                 m_FolderIcon.style.backgroundImage = new StyleBackground(v);
             }
             else
                 m_FolderIcon.SetHiddenFromLayout(true);
         }
     }

     internal class UILayerNameColumn : UIColumn
     {
         PSDImporterLayerManagementMultiColumnTreeView m_TreeView;
         public UILayerNameColumn(PSDImporterLayerManagementMultiColumnTreeView treeView) : base(treeView)
         {
             makeCell = () => new UICellLabelElement();
             bindCell = BindCell;
             sortable = false;
             stretchable = true;
             title = "Layers";
         }

         public virtual void BindCell(VisualElement e, int index)
         {
             var item = treeView.GetFromIndex(index);
             UICellLabelElement label = (UICellLabelElement)e; 
             label.text = item.displayName;
             label.EnableFolderIcon(item.icon);
             label.SetEnabled(!item.disable);
             if (item.disable)
                 label.tooltip = Tooltips.layerHiddenToolTip;
             else
                 label.tooltip = "";
         }
     }
     
}

