!-----------------------------------------------------------------------
  module param00
    implicit none
!-----------------------------------------------------------------------
    integer(kind=4),parameter ::  idata = 2048
    integer(kind=4),parameter ::  th00 = 30   ! WNA [deg]
!-----------------------------------------------------------------------
    real(kind=8),parameter ::  ww   = 0.3d0   ! [fce]
    real(kind=8),parameter ::  fpc  = 4.0d0   ! fp/fce
    real(kind=8),parameter ::  bb   = 5.d-7   ! [nT]
    real(kind=8),parameter ::  dsmp = 1.d0/65.536d3
!-----------------------------------------------------------------------
    real(kind=8),parameter ::  pi = 4.d0*atan(1.d0)
    real(kind=8),parameter ::  cc = 2.99792458d8 , qq = 1.602d-19
    real(kind=8),parameter ::  kb = 1.38d-23 , me = 9.1d-31 , ev = 1.16d4
!-----------------------------------------------------------------------
    integer(kind=4),parameter ::  ipol = +1
!####                             ^^^^ +1 : R-mode    -1 : L-mode
!-----------------------------------------------------------------------
  end module param00
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
  module main00
!-----------------------------------------------------------------------
  contains
!=======================================================================
!=======================================================================
  subroutine main_model_waveform
!-----------------------------------------------------------------------
    use param00
    implicit none
!-----------------------------------------------------------------------
    real(kind=8)    ::  eb(6,idata)  ! 1-3: Ex-Ez  4-6: Bx-Bz  B0=(0,0,1)
    real(kind=8)    ::  tt,Bx_max,By_max,Br0,Bl0,Br(2),Bl(2)
    real(kind=8)    ::  Bx1,By1,bw_perp,ang00,bw_perp_r,ang00_r
    integer(kind=4) ::  ii,jj
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
    open(10,file="waveform.dat",status="unknown")
    open(11,file="waveform_RL.dat",status="unknown")
!-----------------------------------------------------------------------
    call model_waveform(eb)
!-----------------------------------------------------------------------
    Bx_max = 0.d0
    By_max = 0.d0
!-----------------------------------------------------------------------
    Bx_max = maxval(eb(4,:))
    By_max = maxval(eb(5,:))
    print '(2(1PE12.4))',Bx_max,By_max
    Br0 = 0.5d0*(Bx_max + By_max)
    Bl0 = abs(0.5d0*(Bx_max - By_max))
    print '(4(1PE12.4))',Br0,Bl0,Br0/Br0,Bl0/Br0
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
    do ii=1,idata
!-----------------------------------------------------------------------
      tt = dble(ii)*dsmp
      Bx1 = eb(4,ii)
      By1 = eb(5,ii)
      Br(1) =  Bx1*Br0/Bx_max
      Br(2) =  By1*Br0/By_max
      Bl(1) =  Bx1*Bl0/Bx_max
      Bl(2) = -By1*Bl0/By_max
!-----------------------------------------------------------------------
      bw_perp = sqrt( Bx1**2 + By1**2 )
      ang00 = acos(Bx1/bw_perp)
      ang00 = ang00/pi*180.d0
      if( By1.lt.0.d0 ) ang00 = 360.d0 - ang00
!-----------------------------------------------------------------------
      bw_perp_r = sqrt( Br(1)**2 + Br(2)**2 )
      ang00_r = acos(Br(1)/bw_perp_r)
      ang00_r = ang00_r/pi*180.d0
      if( Br(2).lt.0.d0 ) ang00_r = 360.d0 - ang00_r
!-----------------------------------------------------------------------
      write(10,'(7(1PE12.4))') tt,eb(1,ii),eb(2,ii),eb(3,ii), &
&                                 eb(4,ii),eb(5,ii),eb(6,ii)
      write(11,'(7(1PE12.4))') tt,Br(1),Br(2),Bl(1),Bl(2),ang00,ang00_r
!-----------------------------------------------------------------------
    end do
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
    close(10)
    close(11)
    return
!-----------------------------------------------------------------------
  end subroutine main_model_waveform
!=======================================================================
!=======================================================================
  subroutine model_waveform(eb)
!-----------------------------------------------------------------------
    use param00
    implicit none
!-----------------------------------------------------------------------
    real(kind=8),intent(inout) ::  eb(6,idata)
    real(kind=8) ::  ck,ref_n,vph
    real(kind=8) ::  th,Ey_Ex,Ex_Ez,Ey_Ez
    real(kind=8) ::  Ex,Ey,Ez,Ex00,Ey00,Ez00
    real(kind=8) ::  Bx,By,Bz,Bx00,By00,Bz00
    real(kind=8) ::  Ex1,Ey1,Ez1,Bx1,By1,Bz1
    real(kind=8) ::  XX,YY,phs,wce,tt
    real(kind=8) ::  ang,bw_perp,ang00
    integer(kind=4) ::  iflg,ii,jj
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
    wce  = 1.d4   ! fce [Hz]
    wce  = wce*2.d0*pi
!-----------------------------------------------------------------------
    call AH_rout( ck,ref_n,vph,iflg )
    if( iflg.eq.1 ) then
      write(6,'(11Hno solution)')
      stop
    end if
    write(6,'(50(1H-),/,22HInput parameter is ...,/, &
&             3X,1PE12.5,9H [W/Wce] ,1PE12.5,5H [kR])') ww,ck
    write(6,'(22H Phase Velocity     = ,1PE12.5,4H [c],/,&
&             22H Refractive index   = ,1PE12.5,/,50(1H-))') vph,ref_n
!-----------------------------------------------------------------------
    th = th00*pi/180.d0
!-----------------------------------------------------------------------
    XX = (fpc/ww)**2
    YY = 1.d0/ww
!-----------------------------------------------------------------------
    Ey_Ex = XX*YY/(YY**2-1.d0)/(XX/(YY**2-1.d0)+1.d0-ref_n**2)
    Ex_Ez = (XX-1.d0+ref_n**2*sin(th)**2)/(ref_n*sin(th)*cos(th))**2
    Ey_Ez = EY_Ex*Ex_Ez
!-----------------------------------------------------------------------
    Ey_Ex = abs(Ey_Ex)
    Ex_Ez = abs(Ex_Ez)
    Ey_Ez = abs(Ey_Ez)
!-----------------------------------------------------------------------
    Ex00 = 1.d0/ref_n
    Ey00 = Ey_Ex/ref_n
    Ez00 = 1.d0/Ex_Ez/ref_n
!-----------------------------------------------------------------------
    Bx00 = -cos(th)*Ey00
    By00 =  cos(th)*Ex00 - sin(th)*Ez00
    Bz00 =  sin(th)*Ey00
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
    do ii=1,idata
!-----------------------------------------------------------------------
      tt = dble(ii)*dsmp
      phs = ww*wce*tt
      Ex = Ex00*cos(phs)
      Ey = Ey00*sin(phs)
      Ez = Ez00*cos(phs)
      Bx = Bx00*sin(phs)
      By = By00*cos(phs)
      Bz = Bz00*sin(phs)
!-----------------------------------------------------------------------
      bw_perp = sqrt( Bx**2 + By**2 )
      ang = acos(Bx/bw_perp)
      ang = ang/pi*180.d0
      if( By.lt.0.d0 ) ang = 360.d0 - ang
!-----------------------------------------------------------------------
      Ex1 =  Ex*cos(th) - Ez*sin(th)
      Ey1 =  Ey
      Ez1 = -Ex*sin(th) + Ez*cos(th)
      Bx1 =  Bx*cos(th) - Bz*sin(th)
      By1 =  By
      Bz1 =  Bx*sin(th) + Bz*cos(th)
      bw_perp = sqrt( Bx1**2 + By1**2 )
      ang00 = acos(Bx1/bw_perp)
      ang00 = ang00/pi*180.d0
      if( By1.lt.0.d0 ) ang00 = 360.d0 - ang00
!-----------------------------------------------------------------------
      eb(1,ii) = Ex
      eb(2,ii) = Ey
      eb(3,ii) = Ez
      eb(4,ii) = Bx
      eb(5,ii) = By
      eb(6,ii) = Bz
!-----------------------------------------------------------------------
    end do
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
    return
  end subroutine model_waveform
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
  subroutine AH_rout( pk,ref_n,vph,iflg )
!-----------------------------------------------------------------------
    use param00
    implicit none
!-----------------------------------------------------------------------
    real(kind=8),intent(out)   ::  pk,ref_n,vph
    integer(kind=4),intent(out)::  iflg
!-----------------------------------------------------------------------
    real(kind=8)    ::  fpc2,pka,pw2
    real(kind=8)    ::  radi,wc,te,th,vt
    integer(kind=4) ::  ii,jj,kk,ll
!-----------------------------------------------------------------------
!  PW : w/Wce  PK : kR  ( R = Larmor Radius )
!  FP_FC : fp/fc
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
    te = 1.d0*ev
    vt = sqrt(2.d0*kb*te/me)
    wc = qq*bb/me
    radi = vt/wc
    iflg = 0
!-----------------------------------------------------------------------
    fpc2 = fpc**2
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
    th = th00
    if(th.eq.0.d0 ) th = 0.05d0
    if(th.eq.90.d0) th = 89.95d0
    th = th *pi/180.d0
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
    pw2 = ww**2
    if(pw2.eq.fpc2) then
      iflg = 1
      return
    end if
!-----------------------------------------------------------------------
    pka = sin(th)**4/(pw2-fpc2)**2
    pka = pka + 4.d0/pw2*cos(th)**2
    if( pka.lt.0.d0 ) then
      iflg = 1
      return
    end if
    pka = sqrt(pka)
!-----------------------------------------------------------------------
    if( ipol.eq.-1 ) then
!-----------------------------------------------------------------------
      pk = 2.d0 - sin(th)**2/(pw2-fpc2) + pka
      if( pk.eq.0.d0 ) then
        iflg = 1
        return
      end if
!-----------------------------------------------------------------------
      pk = 2.d0*fpc2/pk
      pk = pw2 - pk
      if( pk.lt.0.d0 ) then
        iflg = 1
        return
      end if
      pk = sqrt(pk)
!-----------------------------------------------------------------------
    else if( ipol.eq.+1 ) then
!-----------------------------------------------------------------------
      pk = 2.d0 - sin(th)**2/(pw2-fpc2) - pka
      if( pk.eq.0.d0 ) then
        iflg = 1
        return
      end if
!-----------------------------------------------------------------------
      pk = 2.d0*fpc2/pk
      pk = pw2 - pk
      if( pk.lt.0.d0 ) then
        iflg = 1
        return
      end if
      pk = sqrt(pk)
!-----------------------------------------------------------------------
     else
       iflg = 1
       write(6,*) "N.A.  check initial ww and ipol"
       return
     end if
!-----------------------------------------------------------------------
    vph = ww/pk
    ref_n = pk/ww
!-----------------------------------------------------------------------
    return
  end subroutine AH_rout
!-----------------------------------------------------------------------
  end module main00
!=======================================================================
!=======================================================================
program prog00
!-----------------------------------------------------------------------
  use main00
!-----------------------------------------------------------------------
  call main_model_waveform
  stop
!-----------------------------------------------------------------------
end program prog00
!-----------------------------------------------------------------------
