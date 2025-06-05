using NUnit.Framework;
using NUnit.Framework.Interfaces;
using NUnit.Framework.Internal.Commands;

namespace Tests_16s
{
    public class FasterThan500msAttribute : NUnitAttribute, IWrapTestMethod
    {
        public TestCommand Wrap(TestCommand command)
        {
            return new FasterThan500msCommand(command);
        }
    }
}