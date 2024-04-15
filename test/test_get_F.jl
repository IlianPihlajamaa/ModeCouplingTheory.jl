function test_getF()
    # scalar case
    F = [rand() for i =1:10]
    sol = (F = F,)
    @test get_F(sol) == F
    @test get_F(sol, 2:4) == F[2:4]
    # vector case
    F = [[1,2], [3,4], [5,6]]
    sol = (F = F,)
    @test get_F(sol) == F
    @test get_F(sol, 2, 1) == 3
    @test get_F(sol, 2, 2) == 4
    @test get_F(sol, 2, 1:2) == [3, 4]
    @test get_F(sol, 2:3, 1) == [3, 5]
    @test get_F(sol, 2:3, 1:2) == [[3, 4], [5, 6]]
    # matrix case
    F = [[[1 2]; [3 4]], [[5 6]; [7 8]]]
    sol = (F = F,)
    @test get_F(sol) == F
    @test get_F(sol, 2, (1,1)) == 5
    @test get_F(sol, 2, (1,2)) == 6
    @test get_F(sol, 2, (1,1:2)) == [5, 6]
    @test get_F(sol, 2, (1:2, 1)) == [5, 7]
    @test get_F(sol, 2, (1:2, 1:2)) == [[5 6]; [7 8]]
    @test get_F(sol, :, (1,1)) == [1, 5]
    
    # vector of Smatrix case

    F = [[SMatrix{2,2}(1,2,3,4), SMatrix{2,2}(5,6,7,8)], [SMatrix{2,2}(9,10,11,12), SMatrix{2,2}(13,14,15,16)]]
    sol = (F = F,)
    @test get_F(sol) == F
    @test get_F(sol, 2, 1) == SMatrix{2,2}(9,10,11,12)
    @test get_F(sol, 2, 2) == SMatrix{2,2}(13,14,15,16)
    @test get_F(sol, 2, 1:2) == [SMatrix{2,2}(9,10,11,12), SMatrix{2,2}(13,14,15,16)]
    @test get_F(sol, 1:2, 1) == [SMatrix{2,2}(1,2,3,4), SMatrix{2,2}(9,10,11,12)]
    @test get_F(sol, 1, 2, (1,2)) == 6
    @test get_F(sol, 1, 2, (1,1)) == 5
    @test get_F(sol, 1, 2, (1,1:2)) == [5, 6]
    @test get_F(sol, 1, 2, (1:2, 1)) == [5, 7]
    @test get_F(sol, :, 2, (1,2)) == [7, 15]
    @test get_F(sol, 1, :, (1,1)) == [1, 9]
end

test_getF()
