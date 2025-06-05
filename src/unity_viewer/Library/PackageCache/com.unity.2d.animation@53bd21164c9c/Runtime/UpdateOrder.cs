namespace UnityEngine.U2D.Animation
{
    /// <summary>
    /// Default script update order for 2D Animation systems.
    /// </summary>
    internal static class UpdateOrder
    {
        /// <summary>
        /// Sprite Resolver execution order.
        /// </summary>
        public const int spriteResolverUpdateOrder = -20;

        /// <summary>
        /// IK Manager 2D execution order.
        /// </summary>
        public const int ikUpdateOrder = -10;

        /// <summary>
        /// Sprite Skin execution order.
        /// </summary>
        public const int spriteSkinUpdateOrder = 10;
    }
}
