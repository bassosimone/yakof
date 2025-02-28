"""Tests for the yakof.frontend.abstract module."""

# SPDX-License-Identifier: Apache-2.0

import pytest
from yakof.frontend import abstract, graph


class DummyBasis:
    """Dummy basis type for testing."""

    axes = 0


def test_tensor_creation():
    """Test basic tensor creation and properties."""
    space = abstract.TensorSpace(DummyBasis())

    # Test placeholder creation
    x = space.placeholder("x", 1.0)
    assert x.name == "x"
    assert isinstance(x.node, graph.placeholder)
    assert x.node.default_value == 1.0

    # Test constant creation
    c = space.constant(2.0, "c")
    assert c.name == "c"
    assert isinstance(c.node, graph.constant)
    assert c.node.value == 2.0


def test_tensor_arithmetic():
    """Test arithmetic operations between tensors."""
    space = abstract.TensorSpace(DummyBasis())
    x = space.placeholder("x")
    y = space.placeholder("y")
    c = space.constant(2.0)

    # Test operator overloading
    add1 = x + y
    assert isinstance(add1.node, graph.add)
    assert add1.node.left is x.node
    assert add1.node.right is y.node

    # Test scalar operations
    add2 = x + 2.0
    assert isinstance(add2.node, graph.add)
    assert add2.node.left is x.node
    assert isinstance(add2.node.right, graph.constant)

    # Test reverse operations
    add3 = 2.0 + x
    assert isinstance(add3.node, graph.add)
    assert isinstance(add3.node.left, graph.constant)
    assert add3.node.right is x.node

    # Test other arithmetic operations
    sub = x - y
    assert isinstance(sub.node, graph.subtract)
    mul = x * y
    assert isinstance(mul.node, graph.multiply)
    div = x / y
    assert isinstance(div.node, graph.divide)


def test_tensor_comparisons():
    """Test comparison operations between tensors."""
    space = abstract.TensorSpace(DummyBasis())
    x = space.placeholder("x")
    y = space.placeholder("y")

    # Test all comparison operators
    eq = x == y
    assert isinstance(eq.node, graph.equal)
    ne = x != y
    assert isinstance(ne.node, graph.not_equal)
    lt = x < y
    assert isinstance(lt.node, graph.less)
    le = x <= y
    assert isinstance(le.node, graph.less_equal)
    gt = x > y
    assert isinstance(gt.node, graph.greater)
    ge = x >= y
    assert isinstance(ge.node, graph.greater_equal)


def test_tensor_logical():
    """Test logical operations between tensors."""
    space = abstract.TensorSpace(DummyBasis())
    x = space.placeholder("x")
    y = space.placeholder("y")

    # Test logical operators
    and_op = x & y
    assert isinstance(and_op.node, graph.logical_and)
    or_op = x | y
    assert isinstance(or_op.node, graph.logical_or)
    xor_op = x ^ y
    assert isinstance(xor_op.node, graph.logical_xor)
    not_op = ~x
    assert isinstance(not_op.node, graph.logical_not)


def test_tensor_math():
    """Test mathematical operations on tensors."""
    space = abstract.TensorSpace(DummyBasis())
    x = space.placeholder("x")
    y = space.placeholder("y")

    # Test math operations
    exp = space.exp(x)
    assert isinstance(exp.node, graph.exp)
    log = space.log(x)
    assert isinstance(log.node, graph.log)
    power = space.power(x, y)
    assert isinstance(power.node, graph.power)
    maximum = space.maximum(x, y)
    assert isinstance(maximum.node, graph.maximum)


def test_tensor_conditional():
    """Test conditional operations on tensors."""
    space = abstract.TensorSpace(DummyBasis())
    cond = space.placeholder("cond")
    x = space.placeholder("x")
    y = space.placeholder("y")

    # Test where operation
    where = space.where(cond, x, y)
    assert isinstance(where.node, graph.where)
    assert where.node.condition is cond.node
    assert where.node.then is x.node
    assert where.node.otherwise is y.node

    # Test multi_clause_where
    clauses = [(cond, x)]
    multi_where = space.multi_clause_where(clauses, y)
    assert isinstance(multi_where.node, graph.multi_clause_where)
    assert multi_where.node.clauses[0][0] is cond.node
    assert multi_where.node.clauses[0][1] is x.node
    assert multi_where.node.default_value is y.node


def test_tensor_debug():
    """Test debug operations on tensors."""
    space = abstract.TensorSpace(DummyBasis())
    x = space.placeholder("x")

    # Test tracepoint
    traced = space.tracepoint(x)
    assert traced.node.flags & graph.NODE_FLAG_TRACE
    assert traced.node is x.node

    # Test breakpoint
    broken = space.breakpoint(x)
    assert broken.node.flags & graph.NODE_FLAG_BREAK
    assert broken.node.flags & graph.NODE_FLAG_TRACE
    assert broken.node is x.node


def test_tensor_identity():
    """Test tensor identity and hashing behavior."""
    space = abstract.TensorSpace(DummyBasis())
    x = space.placeholder("x")
    y = space.placeholder("x")  # Same name, different tensor

    # Test identity
    assert x is not y
    assert hash(x) != hash(y)

    # Test dictionary usage
    d = {x: 1, y: 2}
    assert len(d) == 2
    assert d[x] == 1
    assert d[y] == 2


def test_tensor_space_method_consistency():
    """Test that TensorSpace methods are consistent with Tensor operators."""
    space = abstract.TensorSpace(DummyBasis())
    x = space.placeholder("x")
    y = space.placeholder("y")

    # Compare operator results with method results for binary operations
    add_op = x + y
    add_method = space.add(x, y)
    assert isinstance(add_op.node, graph.BinaryOp)
    assert isinstance(add_method.node, graph.BinaryOp)
    assert add_op.node.left is add_method.node.left

    sub_op = x - y
    sub_method = space.subtract(x, y)
    assert isinstance(sub_op.node, graph.BinaryOp)
    assert isinstance(sub_method.node, graph.BinaryOp)
    assert sub_op.node.left is sub_method.node.left

    mul_op = x * y
    mul_method = space.multiply(x, y)
    assert isinstance(mul_op.node, graph.BinaryOp)
    assert isinstance(mul_method.node, graph.BinaryOp)
    assert mul_op.node.left is mul_method.node.left

    div_op = x / y
    div_method = space.divide(x, y)
    assert isinstance(div_op.node, graph.BinaryOp)
    assert isinstance(div_method.node, graph.BinaryOp)
    assert div_op.node.left is div_method.node.left

    eq_op = x == y
    eq_method = space.equal(x, y)
    assert isinstance(eq_op.node, graph.BinaryOp)
    assert isinstance(eq_method.node, graph.BinaryOp)
    assert eq_op.node.left is eq_method.node.left

    ne_op = x != y
    ne_method = space.not_equal(x, y)
    assert isinstance(ne_op.node, graph.BinaryOp)
    assert isinstance(ne_method.node, graph.BinaryOp)
    assert ne_op.node.left is ne_method.node.left

    lt_op = x < y
    lt_method = space.less(x, y)
    assert isinstance(lt_op.node, graph.BinaryOp)
    assert isinstance(lt_method.node, graph.BinaryOp)
    assert lt_op.node.left is lt_method.node.left

    le_op = x <= y
    le_method = space.less_equal(x, y)
    assert isinstance(le_op.node, graph.BinaryOp)
    assert isinstance(le_method.node, graph.BinaryOp)
    assert le_op.node.left is le_method.node.left

    gt_op = x > y
    gt_method = space.greater(x, y)
    assert isinstance(gt_op.node, graph.BinaryOp)
    assert isinstance(gt_method.node, graph.BinaryOp)
    assert gt_op.node.left is gt_method.node.left

    ge_op = x >= y
    ge_method = space.greater_equal(x, y)
    assert isinstance(ge_op.node, graph.BinaryOp)
    assert isinstance(ge_method.node, graph.BinaryOp)
    assert ge_op.node.left is ge_method.node.left

    and_op = x & y
    and_method = space.logical_and(x, y)
    assert isinstance(and_op.node, graph.BinaryOp)
    assert isinstance(and_method.node, graph.BinaryOp)
    assert and_op.node.left is and_method.node.left

    or_op = x | y
    or_method = space.logical_or(x, y)
    assert isinstance(or_op.node, graph.BinaryOp)
    assert isinstance(or_method.node, graph.BinaryOp)
    assert or_op.node.left is or_method.node.left

    xor_op = x ^ y
    xor_method = space.logical_xor(x, y)
    assert isinstance(xor_op.node, graph.BinaryOp)
    assert isinstance(xor_method.node, graph.BinaryOp)
    assert xor_op.node.left is xor_method.node.left

    # For unary operations, check type and operand instead of identity
    not_op = ~x
    not_method = space.logical_not(x)
    assert isinstance(not_op.node, graph.UnaryOp)
    assert isinstance(not_method.node, graph.UnaryOp)
    assert not_op.node.node is x.node
    assert not_method.node.node is x.node


def test_tensor_name_property():
    """Test tensor name getter and setter."""
    space = abstract.TensorSpace(DummyBasis())
    x = space.placeholder("x")

    # Test initial name
    assert x.name == "x"

    # Test name setter
    x.name = "renamed"
    assert x.name == "renamed"
    assert x.node.name == "renamed"  # Verify it propagates to node


def test_tensor_reverse_operations():
    """Test reverse arithmetic and logical operations."""
    space = abstract.TensorSpace(DummyBasis())
    x = space.placeholder("x")

    # Test reverse subtraction (other - x)
    rsub = 2.0 - x
    assert isinstance(rsub.node, graph.subtract)
    assert isinstance(rsub.node.left, graph.constant)
    assert rsub.node.left.value == 2.0
    assert rsub.node.right is x.node

    # Test reverse multiplication (other * x)
    rmul = 2.0 * x
    assert isinstance(rmul.node, graph.multiply)
    assert isinstance(rmul.node.left, graph.constant)
    assert rmul.node.left.value == 2.0
    assert rmul.node.right is x.node

    # Test reverse division (other / x)
    rdiv = 2.0 / x
    assert isinstance(rdiv.node, graph.divide)
    assert isinstance(rdiv.node.left, graph.constant)
    assert rdiv.node.left.value == 2.0
    assert rdiv.node.right is x.node

    # Test reverse logical AND (other & x)
    rand = True & x
    assert isinstance(rand.node, graph.logical_and)
    assert isinstance(rand.node.left, graph.constant)
    assert rand.node.left.value is True
    assert rand.node.right is x.node

    # Test reverse logical OR (other | x)
    ror = True | x
    assert isinstance(ror.node, graph.logical_or)
    assert isinstance(ror.node.left, graph.constant)
    assert ror.node.left.value is True
    assert ror.node.right is x.node

    # Test reverse logical XOR (other ^ x)
    rxor = True ^ x
    assert isinstance(rxor.node, graph.logical_xor)
    assert isinstance(rxor.node.left, graph.constant)
    assert rxor.node.left.value is True
    assert rxor.node.right is x.node


def test_tensor_reverse_operations_with_tensors():
    """Test reverse operations with tensors instead of scalars."""
    space = abstract.TensorSpace(DummyBasis())
    x = space.placeholder("x")
    y = space.placeholder("y")

    # Test reverse subtraction (y - x)
    rsub = y - x
    assert isinstance(rsub.node, graph.subtract)
    assert rsub.node.left is y.node
    assert rsub.node.right is x.node

    # Test reverse multiplication (y * x)
    rmul = y * x
    assert isinstance(rmul.node, graph.multiply)
    assert rmul.node.left is y.node
    assert rmul.node.right is x.node

    # Test reverse division (y / x)
    rdiv = y / x
    assert isinstance(rdiv.node, graph.divide)
    assert rdiv.node.left is y.node
    assert rdiv.node.right is x.node

    # Test reverse logical operations
    rand = y & x
    assert isinstance(rand.node, graph.logical_and)
    assert rand.node.left is y.node
    assert rand.node.right is x.node

    ror = y | x
    assert isinstance(ror.node, graph.logical_or)
    assert ror.node.left is y.node
    assert ror.node.right is x.node

    rxor = y ^ x
    assert isinstance(rxor.node, graph.logical_xor)
    assert rxor.node.left is y.node
    assert rxor.node.right is x.node


def test_ensure_same_basis():
    """Test ensure_same_basis function."""
    # Same instance - should pass
    basis1 = DummyBasis()
    abstract.ensure_same_basis(basis1, basis1)

    # Different instances but same axes - should pass
    class TestBasis:
        axes = (1, 2, 3)

    basis2 = TestBasis()
    basis3 = TestBasis()
    abstract.ensure_same_basis(basis2, basis3)

    # Different axes - should fail
    class AnotherBasis:
        axes = (4, 5, 6)

    basis4 = AnotherBasis()
    with pytest.raises(ValueError, match="Tensors must have the same basis"):
        abstract.ensure_same_basis(basis2, basis4)

    # Non-Basis object - should fail
    not_a_basis = "not a basis"
    with pytest.raises(TypeError):
        abstract.ensure_same_basis(basis1, not_a_basis)

    with pytest.raises(TypeError):
        abstract.ensure_same_basis(not_a_basis, basis1)


def test_tensor_operations_with_different_bases():
    """Test operations between tensors with different bases."""

    # Create two spaces with different bases
    class BasisA:
        axes = (1, 2)

    class BasisB:
        axes = (3, 4)

    space_a = abstract.TensorSpace(BasisA())
    space_b = abstract.TensorSpace(BasisB())

    x = space_a.placeholder("x")
    y = space_b.placeholder("y")

    # Attempting operations between tensors with different bases should fail
    with pytest.raises(ValueError, match="Tensors must have the same basis"):
        x + y  # type: ignore

    with pytest.raises(ValueError, match="Tensors must have the same basis"):
        space_a.add(x, y)  # type: ignore

    # Test other operations
    with pytest.raises(ValueError):
        x - y  # type: ignore

    with pytest.raises(ValueError):
        x * y  # type: ignore

    with pytest.raises(ValueError):
        x / y  # type: ignore

    with pytest.raises(ValueError):
        x == y  # type: ignore

    with pytest.raises(ValueError):
        x < y  # type: ignore

    with pytest.raises(ValueError):
        x & y  # type: ignore


def test_tensor_math_with_different_bases():
    """Test mathematical operations between tensors with different bases."""

    # Create two spaces with different bases
    class BasisA:
        axes = (1, 2)

    class BasisB:
        axes = (3, 4)

    space_a = abstract.TensorSpace(BasisA())
    space_b = abstract.TensorSpace(BasisB())

    x = space_a.placeholder("x")
    y = space_b.placeholder("y")

    # Test math operations with different bases
    with pytest.raises(ValueError):
        space_a.power(x, y)  # type: ignore

    with pytest.raises(ValueError):
        space_a.maximum(x, y)  # type: ignore

    with pytest.raises(ValueError):
        space_a.where(x, y, y)  # type: ignore

    # This should also fail because then/otherwise have different bases
    with pytest.raises(ValueError):
        space_a.where(x, x, y)  # type: ignore


def test_tensor_space_axes_method():
    """Test that TensorSpace.axes() raises TypeError for invalid basis."""

    # Valid case - basis implements Basis protocol
    class ValidBasis:
        axes = (1, 2, 3)

    valid_space = abstract.TensorSpace(ValidBasis())
    assert valid_space.axes() == (1, 2, 3)

    # Invalid case - basis doesn't implement Basis protocol
    class InvalidBasis:
        # Missing axes attribute
        pass

    invalid_space = abstract.TensorSpace(InvalidBasis())
    with pytest.raises(TypeError, match="must be an instance of Basis"):
        invalid_space.axes()

    # Another invalid case - basis is not an object
    string_space = abstract.TensorSpace("not an object with axes")
    with pytest.raises(TypeError, match="must be an instance of Basis"):
        string_space.axes()

    # None as basis
    none_space = abstract.TensorSpace(None)
    with pytest.raises(TypeError, match="must be an instance of Basis"):
        none_space.axes()
