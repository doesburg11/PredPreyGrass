using System;

namespace UnityEditor.Tilemaps
{
    internal class GridPaintCreateTargetsDropdown : FlexibleMenu
    {
        public static bool IsOpen = false;

        public GridPaintCreateTargetsDropdown(IFlexibleMenuItemProvider itemProvider, int selectionIndex, FlexibleMenuModifyItemUI modifyItemUi, Action<int, object> itemClickedCallback, float minWidth)
            : base(itemProvider, selectionIndex, modifyItemUi, itemClickedCallback)
        {
            minTextWidth = minWidth;
        }

        public override void OnOpen()
        {
            base.OnOpen();
            IsOpen = true;
        }

        public override void OnClose()
        {
            IsOpen = false;
            base.OnClose();
        }

        internal class MenuItemProvider : IFlexibleMenuItemProvider
        {
            private bool m_HasTilePalette;

            public MenuItemProvider(bool hasTilePalette)
            {
                m_HasTilePalette = hasTilePalette;
            }
            
            public int Count()
            {
                return GameObjectCreation.CreateTilemapTargetsNames.Length - (m_HasTilePalette ? 0 : 1);
            }

            public object GetItem(int index)
            {
                if (!m_HasTilePalette)
                    index += 1;
                if (index < GameObjectCreation.CreateTilemapTargetsNames.Length)
                    return GameObjectCreation.CreateTilemapTargetsNames[index];
                return null;
            }

            public int Add(object obj)
            {
                throw new NotImplementedException();
            }

            public void Replace(int index, object newPresetObject)
            {
                throw new NotImplementedException();
            }

            public void Remove(int index)
            {
                throw new NotImplementedException();
            }

            public object Create()
            {
                throw new NotImplementedException();
            }

            public void Move(int index, int destIndex, bool insertAfterDestIndex)
            {
                throw new NotImplementedException();
            }

            public string GetName(int index)
            {
                if (!m_HasTilePalette)
                    index += 1;
                if (index < GameObjectCreation.CreateTilemapTargetsNames.Length)
                    return GameObjectCreation.CreateTilemapTargetsNames[index];
                return "";
            }

            public bool IsModificationAllowed(int index)
            {
                return false;
            }

            public int[] GetSeperatorIndices()
            {
                return Array.Empty<int>();
            }
        }
    }
}
