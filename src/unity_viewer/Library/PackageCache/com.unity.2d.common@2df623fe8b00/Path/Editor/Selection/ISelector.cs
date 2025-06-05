namespace UnityEditor.U2D.Common.Path
{
    internal interface ISelector<T>
    {
        bool Select(T element);
    }
}
