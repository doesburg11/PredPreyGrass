namespace UnityEditor.U2D.Common.Path
{
    internal interface ISelectable<T>
    {
        bool Select(ISelector<T> selector);
    }
}
