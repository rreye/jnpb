! Author: Claude (Anthropic)
! Reference driver for NPB.common.numerics.Random (see gen_random_golden.py).
! It is linked with common/randdp.f90 (or randi8.f90) and with the official routines
! power (OMP MG), power4 (MPI MG, n1*n2+n3 form) and ipow46 (FT), which the script
! extracts from the official sources and renames (power2, power4) to avoid clashes.
! Output: one line per datum; the script turns them into a .properties file.
      program random_ref
      implicit none
      double precision randlc, power2, power4
      external randlc, power2, power4
      double precision x, a, r, y(1000), x0s(5), as(5), pa(3), res, res2
      integer c, i, k, n, n1, n2, n3, ns(23), t4(3,14)
      integer(kind=8) big

      x0s = (/ 314159265.d0, 271828183.d0, 70368744177663.d0, 35184372088833.d0, 1234567891.d0 /)
      as  = (/ 1220703125.d0, 1220703125.d0, 70368744177661.d0, 17592186044417.d0, 33554431.d0 /)

      do c = 1, 5
         x = x0s(c)
         a = as(c)
         write(*,'(a,i0,1x,i0,1x,i0)') 'CASE ', c, int(x0s(c),8), int(as(c),8)
         do i = 1, 1000
            r = randlc(x, a)
            write(*,'(a,i0,1x,i0,1x,i0,1x,z16.16)') 'SEQ ', c, i, int(x,8), transfer(r, 0_8)
         end do
         x = x0s(c)
         call vranlc(1000, x, a, y)
         do i = 1, 1000
            write(*,'(a,i0,1x,i0,1x,z16.16)') 'VRN ', c, i, transfer(y(i), 0_8)
         end do
         write(*,'(a,i0,1x,i0)') 'VRNX ', c, int(x,8)
      end do

      x = x0s(1)
      a = as(1)
      do i = 1, 1000000
         r = randlc(x, a)
         if (mod(i,10000) .eq. 0) write(*,'(a,i0,1x,i0)') 'LONG ', i, int(x,8)
      end do

      pa = (/ 1220703125.d0, 314159265.d0, 70368744177661.d0 /)
      ns = (/ 0, 1, 2, 3, 4, 5, 7, 10, 31, 32, 64, 100, 255, 1000, 4096, 123456, &
     &        1048576, 16777217, 33554432, 262144*1024, 987654321, 1073741823, 1073741824 /)
      do k = 1, 3
         do i = 1, 23
            a = pa(k)
            n = ns(i)
            res = power2(a, n)
            call ipow46(a, n, res2)
            write(*,'(a,i0,1x,i0,1x,i0,1x,i0)') 'POW ', int(a,8), n, int(res,8), int(res2,8)
         end do
      end do

      t4 = reshape( (/ &
     &   512,1,0,  512,512,0,  256,1,0,  256,256,0,  1024,1,0,  1024,1024,0, &
     &   512,3,1,  512,100,7,  256,300,45,  64,5,0,  7,11,13,  1,1,1, &
     &   1024,1000,999,  4,2,0 /), (/ 3, 14 /) )
      a = as(1)
      do i = 1, 14
         n1 = t4(1,i)
         n2 = t4(2,i)
         n3 = t4(3,i)
         res = power4(a, n1, n2, n3)
         big = int(n1,8)*int(n2,8) + int(n3,8)
         if (big .lt. 2147483647_8) then
            res2 = power2(a, int(big))
         else
            res2 = -1.d0
         end if
         write(*,'(a,i0,1x,i0,1x,i0,1x,i0,1x,i0,1x,i0)') 'POW4 ', int(a,8), n1, n2, n3, int(res,8), int(res2,8)
      end do
      end program random_ref
