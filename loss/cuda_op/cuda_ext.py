import sort_vertices
from torch.autograd import Function

class SortVertices(Function):
    @staticmethod
    def forward(ctx, vertices, mask, num_valid):
        idx = sort_vertices.sort_vertices_forward(vertices, mask, num_valid)
        ctx.mark_non_differentiable(idx)
        return idx
    
    @staticmethod
    def backward(ctx, gradout):
        return ()

sort_v = SortVertices.apply
