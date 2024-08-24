# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 17:46:34 2023
Weighted CNN作为光滑化算子; 输入为元胞平均值（无重复）
@author: 25326
"""

import torch
torch.set_printoptions(precision=6) # Set displayed output precision to 10 digits
import numpy as np
import matplotlib.pyplot as plt
from time import time

class PDENN(torch.nn.Module):
    '''FNN for PDE'''
    def __init__(self, dims):
        super(PDENN, self).__init__()
        self.dims = dims
        for i in range(len(dims[:-1])):
            exec('self.linear{} = torch.nn.Linear(dims[{}], dims[{}])'\
                 .format(i,i,i+1))
            if i<len(dims[:-1])-1:
                exec('self.nonlinear{} = torch.nn.Tanh()'.format(i))
        
        
    def forward(self, xt):
        self.u=xt
        for i in range(len(self.dims[:-1])-1):
            exec('self.u = self.nonlinear{}(self.linear{}(self.u))'.format(i,i))
        exec('self.u = self.linear{}(self.u)'.format(len(self.dims[:-1])-1))
        return self.u
    
class PDECNN1(torch.nn.Module):
    '''Weighted CNN for PDE'''
    def __init__(self, n_ch):
        super(PDECNN1, self).__init__()
        self.n_ch = n_ch
        self.check = 1
        self.conv1 = torch.nn.Conv1d(1,n_ch,k_s[0],padding=0)
        self.conv2 = torch.nn.Conv1d(n_ch,n_ch,k_s[1],padding=0)        
        self.conv3 = torch.nn.Conv1d(n_ch,n_ch,k_s[2],padding=0)
        self.conv4 = torch.nn.Conv1d(n_ch,kk*num4,k_s[3],padding=0)
        
        self.field_n = kk1
        self.conv51_ = torch.nn.Conv1d(kk*num4,num4,kk,padding=0)
        self.conv52_ = torch.nn.Conv1d(1,1,kk,padding=0)
        self.conv51_.weight.data=torch.kron(torch.eye(num4)[:,:,None],\
                                            torch.eye(kk)[None,:,:]/kk)
        self.conv51_.weight.requires_grad_(False)
        self.conv51_.bias.data*=0
        self.conv51_.bias.requires_grad_(False)
        self.conv52_.weight.data=torch.ones(1,1,kk)/kk
        self.conv52_.weight.requires_grad_(False)
        self.conv52_.bias.data*=0
        self.conv52_.bias.requires_grad_(False)
        
        self.pool1 = torch.nn.AvgPool1d(3,stride=1,padding=1)
        self.pool2 = torch.nn.MaxPool1d(5,stride=1,padding=2)
        self.pool3 = torch.nn.AvgPool1d(3,stride=1,padding=1)
        self.c1 = torch.nn.Parameter(torch.rand(1,n_ch,1,requires_grad=True))
        self.c2 = torch.nn.Parameter(torch.rand(1,3*n_ch,1,requires_grad=True))
        self.c3 = torch.nn.Parameter(torch.rand(1,n_ch,1,requires_grad=True))


        self.nl1 = torch.nn.Tanh()
        # self.nl2 = torch.nn.PReLU()
        self.nl2 = torch.nn.Tanh()
        self.nl3 = torch.nn.Tanh()
        self.nl4 = torch.nn.Tanhshrink()
                
    def forward(self, xt):
        xt1=torch.einsum('lk,ijk->ijl',Average,xt)
        xt11=torch.einsum('lk,ijk->ijl',Average1,xt)
        mean=xt1.mean(dim=-1,keepdims=True)
        var=xt1.var(dim=-1,keepdims=True)+1e-8
        xt1=(xt1-mean)/var**0.5 #normalization
        xt11=(xt11-mean)/var**0.5
        xt_=(xt-mean)/var**0.5
        minlimit=torch.cat([torch.einsum('lk,ijk->ijl',d1,xt_).abs(),\
                            torch.einsum('lk,ijk->ijl',d2,xt_).abs()],1)\
            .min(1,keepdims=True)[0]
        # minlimit1=self.assign(minlimit)
        
        xt1=torch.cat([2*xt1[:,:,0:1]-xt1[:,:,list(range(self.field_n,0,-1))], xt1, \
                2*xt1[:,:,-1:]-xt1[:,:,list(range(-2,-self.field_n-2,-1))]],-1)
        u=self.nl1(self.conv1(xt1))
        u=self.nl2(self.conv2(u))
        u=self.nl3(self.conv3(u))
        u=self.conv4(u)
        
        # u=minlimit1*torch.tanh(u/(minlimit1+1e-8)) #强制限制
        # u=torch.cat([u.abs()[None,:,:,:],minlimit1[None,:,:,:]],0).min(0)[0]\
        #     *u.sgn()  #强制限制
        # self.w1=torch.einsum('ij,lmj->lmi',A1,\
        #                       ((xt1[:,:,1:]-xt1[:,:,:-1])).pow(2))
        # self.w2=torch.einsum('ij,lmj->lmi',A2,(torch.einsum('ij,lmj->lmi',D_21,xt1))\
        #                   .pow(2)) #计算二阶差分
        # w=1/(self.w1+self.w2+1e-6)
        # w=1/(torch.einsum('ijk,lkm,ijm->ijl',xt1,smooth_i1,xt1).abs()+1e-2)
        w=torch.cat([torch.einsum('ijk,km,ijm->ij',xt1[:,:,i:i+kk],smooth_ii,
            xt1[:,:,i:i+kk]).abs()[:,:,None] for i in range(xt1.shape[-1]-kk+1)],-1)
        w=1/(w+1e-1)
        wu=w*u
        u=self.conv51_(wu)/self.conv52_(w)
        u=u.permute(0,2,1).reshape(-1,1,num4*n_cell)
        if self.check==1:
            u=u-torch.einsum('lk,ijk->ijl',Average1,u)
            u=torch.cat([u.abs()[None,:,:,:],minlimit[None,:,:,:]],0).min(0)[0]\
                *u.sgn()  #强制限制
        # else:
        #     u=torch.cat([u[:,[i+kk*j for j in range(num)],i:i+n_cell]\
        #         .permute(0,2,1).reshape(-1,1,num*n_cell) for i in range(kk)],1)
        u=u-torch.einsum('lk,ijk->ijl',Average1,u) + xt11
        u=torch.einsum('lk,ijk->ijl',evl,u)
        u=u*var**0.5+mean #un-normalization
        return u
    
    def assign(self, minlimit):
        shape=minlimit.shape
        minlimit1=torch.cat(([minlimit]+[torch.zeros(shape[0],shape[1],kk)])\
           *self.field_n+[minlimit], -1).reshape(shape[0],-1,shape[2]+self.field_n)
        minlimit1=minlimit1.repeat(1,num,1)
        return minlimit1

class PDECNN2(torch.nn.Module):
    '''CNN for PDE'''
    def __init__(self, n_ch):
        super(PDECNN2, self).__init__()
        self.n_ch = n_ch
        self.conv1 = torch.nn.Conv1d(1,n_ch,3,padding=1,padding_mode='replicate')
        self.conv2 = torch.nn.Conv1d(n_ch,n_ch,3,padding=1,padding_mode='replicate')        
        self.conv3 = torch.nn.Conv1d(n_ch,n_ch,3,padding=1,padding_mode='replicate')
        self.conv4 = torch.nn.Conv1d(n_ch,num,3,padding=1,padding_mode='replicate')
        
        self.pool1 = torch.nn.AvgPool1d(3,stride=1,padding=1)
        self.pool2 = torch.nn.MaxPool1d(5,stride=1,padding=2)
        self.pool3 = torch.nn.AvgPool1d(3,stride=1,padding=1)
        self.c1 = torch.nn.Parameter(torch.rand(1,n_ch,1,requires_grad=True))
        self.c2 = torch.nn.Parameter(torch.rand(1,3*n_ch,1,requires_grad=True))
        self.c3 = torch.nn.Parameter(torch.rand(1,n_ch,1,requires_grad=True))

        self.nl1 = torch.nn.Tanh()
        # self.nl2 = torch.nn.PReLU()
        self.nl2 = torch.nn.Tanh()
        self.nl3 = torch.nn.Tanh()
        self.nl4 = torch.nn.Tanh()
        
    def f(self, xt, c):
        return torch.exp(-xt**2/4)*xt*c
        
    def forward(self, xt):
        xt1=torch.einsum('lk,ijk->ijl',Average,xt)
        xt11=torch.einsum('lk,ijk->ijl',Average1,xt)
        mean=xt.mean(dim=-1,keepdims=True)
        var=xt.var(dim=-1,keepdims=True)+1e-8
        xt_=(xt-mean)/var**0.5
        xt1=(xt1-mean)/var**0.5
        xt11=(xt11-mean)/var**0.5
        minlimit=torch.cat([torch.einsum('lk,ijk->ijl',d1,xt_).abs(),\
                            torch.einsum('lk,ijk->ijl',d2,xt_).abs()],1)\
            .min(1,keepdims=True)[0]
        
        u=self.nl1(self.conv1(xt1))
        u=self.nl2(self.conv2(u))
        u=self.nl3(self.conv3(u))
        u=self.conv4(u)
        
        u=u.permute(0,2,1).reshape(-1,1,num*n_cell)
        u=u-torch.einsum('lk,ijk->ijl',Average1,u)
        u=torch.cat([u.abs(),minlimit],1).min(1,keepdims=True)[0]*u.sgn() #强制限制
        u=u-torch.einsum('lk,ijk->ijl',Average1,u)+xt11
        u=u*var**0.5+mean
        return u

class PDECNN3(torch.nn.Module):
    '''Weighted CNN for PDE (only in the vicinity of the oscillation point)'''
    def __init__(self, n_ch):
        super(PDECNN3, self).__init__()
        self.n_ch = n_ch
        self.check = 1
        self.conv1 = torch.nn.Conv1d(1,n_ch,k_s[0],padding=0)
        self.conv2 = torch.nn.Conv1d(n_ch,n_ch,k_s[1],padding=0)        
        self.conv3 = torch.nn.Conv1d(n_ch,n_ch,k_s[2],padding=0)
        self.conv4 = torch.nn.Conv1d(n_ch,kk*num,k_s[3],padding=0)
        
        self.field_n = kk1
        self.conv51_ = torch.nn.Conv1d(kk*num,num,kk,padding=0)
        self.conv52_ = torch.nn.Conv1d(1,1,kk,padding=0)
        self.conv51_.weight.data=torch.kron(torch.eye(num)[:,:,None],\
                                            torch.eye(kk)[None,:,:]/kk)
        self.conv51_.weight.requires_grad_(False)
        self.conv51_.bias.data*=0
        self.conv51_.bias.requires_grad_(False)
        self.conv52_.weight.data=torch.ones(1,1,kk)/kk
        self.conv52_.weight.requires_grad_(False)
        self.conv52_.bias.data*=0
        self.conv52_.bias.requires_grad_(False)
        
        self.pool1 = torch.nn.AvgPool1d(3,stride=1,padding=1)
        self.pool2 = torch.nn.MaxPool1d(5,stride=1,padding=2)
        self.pool3 = torch.nn.AvgPool1d(3,stride=1,padding=1)
        self.c1 = torch.nn.Parameter(torch.rand(1,n_ch,1,requires_grad=True))
        self.c2 = torch.nn.Parameter(torch.rand(1,3*n_ch,1,requires_grad=True))
        self.c3 = torch.nn.Parameter(torch.rand(1,n_ch,1,requires_grad=True))

        self.nl1 = torch.nn.Tanh()
        # self.nl2 = torch.nn.PReLU()
        self.nl2 = torch.nn.Tanh()
        self.nl3 = torch.nn.Tanh()
        self.nl4 = torch.nn.Tanhshrink()
            
    def forward(self, xt):
        xt1=torch.einsum('lk,ijk->ijl',Average,xt)
        xt11=torch.einsum('lk,ijk->ijl',Average1,xt)
        xt2=torch.einsum('lk,ijk->ijl',d1,xt)
        xt3=torch.einsum('lk,ijk->ijl',d2,xt)
        xt4=torch.einsum('lk,ijk->ijl',d3,xt)
        xt5=torch.einsum('lk,ijk->ijl',d4,xt)
        ind3=((((xt2+1e-4).sign()+(xt3+1e-4).sign()+\
                (xt4+1e-4).sign()+(xt5+1e-4).sign()).abs()-2.5).sign()+1)/2
        ind4=(((xt2.abs()-xt4.abs()+1e-6).sign()+\
                (xt2.abs()-xt5.abs()+1e-6).sign()+\
                (xt3.abs()-xt4.abs()+1e-6).sign()+\
                (xt3.abs()-xt5.abs()+1e-6).sign()-3.5).sign()+1)/2
        ind5=1-ind3*ind4
        ind5=self.d(ind5)
        mean=xt1.mean(dim=-1,keepdims=True)
        var=xt1.var(dim=-1,keepdims=True)+1e-8
        xt1=(xt1-mean)/var**0.5 #normalization
        xt11=(xt11-mean)/var**0.5
        xt_=(xt-mean)/var**0.5
        minlimit=torch.cat([torch.einsum('lk,ijk->ijl',d1,xt_).abs(),\
                            torch.einsum('lk,ijk->ijl',d2,xt_).abs(),\
                            torch.einsum('lk,ijk->ijl',d1+d2,xt_).abs()],1)\
            .min(1,keepdims=True)[0]
        
        xt1=torch.cat([2*xt1[:,:,0:1]-xt1[:,:,list(range(self.field_n,0,-1))], xt1, \
                2*xt1[:,:,-1:]-xt1[:,:,list(range(-2,-self.field_n-2,-1))]],-1)
        u=self.nl1(self.conv1(xt1))
        u=self.nl2(self.conv2(u))
        u=self.nl3(self.conv3(u))
        u=self.conv4(u)
        
        # self.w1=torch.einsum('ij,lmj->lmi',A1,\
        #     torch.cat([(xt1[:,:,1:]-xt1[:,:,:-1])[None,:,:,:],\
        #                 (x[1::num][1:,0]-x[1::num][:-1,0])\
        #     .repeat(1,xt1.shape[0],xt1.shape[1],1)],0).norm(dim=0).pow(2))#计算线长
        self.w1=torch.einsum('ij,lmj->lmi',A1,\
                              ((xt1[:,:,1:]-xt1[:,:,:-1])).pow(2))
        self.w2=torch.einsum('ij,lmj->lmi',A2,(torch.einsum('ij,lmj->lmi',D_21,xt1))\
                          .pow(2)) #计算二阶差分
        w=1/(self.w1+self.w2+1e-6)
        # w=1/(torch.einsum('ijk,lkm,ijm->ijl',xt1,smooth_i1,xt1).abs()+1e-5)
        wu=w*u
        u=self.conv51_(wu)/self.conv52_(w)
        
        u=u.permute(0,2,1).reshape(-1,1,num*n_cell)   
        if self.check==1:
            u=u-torch.einsum('lk,ijk->ijl',Average1,u)
            u=torch.cat([u.abs()[None,:,:,:],minlimit[None,:,:,:]],0).min(0)[0]\
                *u.sgn()  #强制限制         
        # u=u-torch.einsum('lk,ijk->ijl',Average1,u)
        # u=minlimit*torch.tanh(u) #强制限制
        u=u-torch.einsum('lk,ijk->ijl',Average1,u)+xt11
        u=u*var**0.5+mean #un-normalization
        u=u*ind5+xt*(1-ind5)
        return u
    
    def f(self, xt):
        return torch.sin(xt)
    
    def d(self, ind):
        """连续化"""
        l=[]    
        for i in range(1,7):
            l.append(torch.cat([ind[:,:,num*i:],ind[:,:,0:num*i]*0],2)[:,:,:,None]\
                     *(1-i/7))
            l.append(torch.cat([ind[:,:,-num*i:]*0,ind[:,:,:-num*i]],2)[:,:,:,None]\
                     *(1-i/7))
        d=torch.cat([ind[:,:,:,None]]+l,3).max(dim=3)[0]
        return d

def derivative(outputs,inputs):
    l=outputs.shape[1]
    D=[]
    for i in range(l):
        D.append(torch.autograd.grad(outputs.sum(0)[i],inputs,retain_graph=True,\
                                     create_graph=True)[0])
        # D.append(torch.autograd.grad(outputs.sum(0)[i],inputs,#retain_graph=True,
        #                              create_graph=True)[0])
    D=torch.cat(D,dim=1)
    return D

#%% 
g=9.81;r=0.9 #r=rho1/rho2
def B(xi):
    return torch.zeros_like(xi)
def G(U,xi):
    g1=U[:,0:1]*U[:,1:2]
    g2=0.5*U[:,1:2]**2 + g*(B(xi)+U[:,0:1]+U[:,2:3])
    g3=U[:,2:3]*U[:,3:4]
    g4=0.5*U[:,3:4]**2 + g*(B(xi)+r*U[:,0:1]+U[:,2:3])
    return torch.cat([g1,g2,g3,g4],-1)
def Partial_G(U):
    g_11=U[:,None,1:2]
    g_12=U[:,None,0:1]
    g_13=torch.zeros_like(g_11)
    g_14=torch.zeros_like(g_11)
    g_1=torch.cat([g_11,g_12,g_13,g_14],-1)
    
    g_21=g*torch.ones_like(g_11)
    g_22=U[:,None,1:2]
    g_23=g_21
    g_24=torch.zeros_like(g_11)
    g_2=torch.cat([g_21,g_22,g_23,g_24],-1)
    
    g_31=torch.zeros_like(g_11)
    g_32=torch.zeros_like(g_11)
    g_33=U[:,None,3:4]
    g_34=U[:,None,2:3]
    g_3=torch.cat([g_31,g_32,g_33,g_34],-1)
    
    g_41=g*r*torch.ones_like(g_11)
    g_42=torch.zeros_like(g_11)
    g_43=g*torch.ones_like(g_11)
    g_44=U[:,None,3:4]
    g_4=torch.cat([g_41,g_42,g_43,g_44],-1)
    
    partial_g=torch.cat([g_1,g_2,g_3,g_4],1)
    return partial_g

#%% 定义函数与矩阵
from sets import lgP, dLgP, gLLNodesAndWeights, LagP, dLagP
import sympy as sp
from scipy.optimize import root
from numba import jit,njit

# from torchquad import set_up_backend  # Necessary to enable GPU support
# from torchquad import Trapezoid, Simpson, Boole, MonteCarlo, VEGAS # The available integrators
# from torchquad.utils.set_precision import set_precision
# import torchquad

Nc=4
x_ran=np.array([0,1])
t_ran=np.array([0,0.4])
deg=2
n_cell=400
dx=(x_ran[1]-x_ran[0])/n_cell
num=deg+1
x_s=np.linspace(x_ran[0],x_ran[1],n_cell+1)

# nodes=np.array([-1,0,1])
nodes=gLLNodesAndWeights(num)[0]

###################################设置初始解(系数)##############################
x=np.concatenate([nodes.reshape(-1,1)*dx/deg+i-dx/deg for i in x_s[1:]],0)
x.sort(0)
U=np.zeros([(num)*n_cell,4])
# U[:,0:1]=np.exp(-64*(x-(x_ran[0]+x_ran[1])*0.5)**2)+1
U[:,0:1]=np.concatenate([np.ones_like(x[:num*(n_cell//2)]),
                          np.zeros_like(x[num*(n_cell//2):])+1e-8],0)*2
U[:,1:2]=0*np.sin(x)
# U[:,2:3]=0*x+1
U[:,2:3]=np.concatenate([np.zeros_like(x[:num*(n_cell//2)]),
                          np.ones_like(x[num*(n_cell//2):])],0)*2
U[:,3:4]=0*np.sin(x)
###############################################################################

###########################################################################
k_s=[3,3,3,3]  #kernel size
kk=sum(k_s)-3 #Receptive field size
kk1=kk-1
D_21=torch.zeros(n_cell+2*kk1,n_cell+2*kk1)
for i in range(1,n_cell+2*kk1-1):
    D_21[i,i-1:i+2]=torch.tensor([1,-2,1])
D_21[0,:]=D_21[1,:];D_21[-1]=D_21[-2,:]

A1=torch.zeros(n_cell+kk1,n_cell-1+2*kk1)
for j in range(n_cell+kk1):
    A1[j,j:j+kk1]=torch.tensor([1/kk1]*kk1)

A2=torch.zeros(n_cell+kk1,n_cell+2*kk1)
for j in range(n_cell+kk1):
    A2[j,j+1:j+kk1]=torch.tensor([1/(kk1-1)]*(kk1-1))

# 使用shu chi-wang论文中的光滑性度量作为权重
t=time()
kk2=3
xi=sp.symbols('xi')
smooth_i=np.zeros([kk2-1,kk2,kk2])
nodes1=np.array(range(-int((kk2-1)/2),int((kk2-1)/2)+1))
for l in range(1,kk2):
    for i in range(kk2):     #M_ij=int(l_i(x)l_j(x))
        for j in range(kk2):
            smooth_i[l-1,i,j]=sp.lambdify(xi,
                    sp.integrate(sp.diff(LagP(i,nodes1,xi),xi,l)\
                    *sp.diff(LagP(j,nodes1,xi),xi,l),(xi,nodes1[0],nodes1[-1]))\
                        ,'numpy')(0)
print('time of generating matrix:{} s'.format(time()-t))

np.save('./smooth_i.npy',smooth_i)
smooth_i=torch.from_numpy(np.load('./smooth_i.npy'))
smooth_ii=torch.zeros(kk,kk)
for i in range(kk-kk2+1):
    smooth_ii[i:i+kk2,i:i+kk2]=smooth_ii[i:i+kk2,i:i+kk2]+smooth_i.sum(0)
smooth_i1=torch.zeros(n_cell+kk1,n_cell+2*kk1,n_cell+2*kk1)
for i in range(n_cell+kk1):
    smooth_i1[i,i:i+kk,i:i+kk]=smooth_ii
###########################################################################

#计算刚度矩阵和质量矩阵
xi=sp.symbols('xi')
int_f=np.zeros(num)
for i in range(num):     #计算基函数的积分
    int_f[i]=sp.lambdify(xi,sp.integrate(LagP(i,nodes,xi),(xi,-1,1)),'numpy')(0)
M=np.zeros([num,num])
for i in range(num):     #M_ij=int(l_i(x)l_j(x))
    for j in range(num):
        M[i,j]=sp.lambdify(xi,
               sp.integrate(LagP(i,nodes,xi)*LagP(j,nodes,xi),(xi,-1,1)),'numpy')(0)
S=np.zeros([num,num])
for i in range(num):     #S_ij=int(l_i_x(x)l_j(x))
    for j in range(num):
        S[i,j]=sp.lambdify(xi,
               sp.integrate(dLagP(i,nodes,xi)*LagP(j,nodes,xi),(xi,-1,1)),'numpy')(0)
# for i in range(num):     #S_ij=int(l_i(x)l_j_x(x))
#     for j in range(num):
#         S[i,j]=sp.lambdify(xi,
#                sp.integrate(LagP(i,nodes,xi)*sp.diff(LagP(j,nodes,xi),xi),(xi,-1,1)),'numpy')(0)
Mk=M*dx/2

#设置中间系数矩阵
MK=np.zeros([(num)*n_cell,(num)*n_cell])
SK=np.zeros([(num)*n_cell,(num)*n_cell])
Fg=np.zeros([(num)*n_cell,(num)*n_cell]);Fg[0,0],Fg[-1,-1]=1,-1
Fg1=np.zeros([(num)*n_cell,(num)*n_cell])
Fu=np.eye((num)*n_cell)
Fu1=np.eye((num)*n_cell)
Fu2=np.eye((num)*n_cell)
Average=np.zeros([n_cell,(num)*n_cell])
Average1=np.zeros([(num)*n_cell,(num)*n_cell])
Average2=torch.zeros([(num)*n_cell,(num)*n_cell])
block1=torch.tensor([[1/4,1/4,0,-1/4,-1/4],[0,0,0,0,0],[-1/4,-1/4,0,1/4,1/4]])
block2=torch.tensor([[2/4,0,-1/4,-1/4],[0,0,0,0],[-2/4,0,1/4,1/4]])
block3=torch.tensor([[1/4,1/4,0,-2/4],[0,0,0,0],[-1/4,-1/4,0,2/4]])
Average2[:3,:4]=block2;Average2[-3:,-4:]=block3
for i in range(n_cell):
    MK[num*i:num*i+num,num*i:num*i+num]=Mk
    SK[num*i:num*i+num,num*i:num*i+num]=S
    Average[i,num*i:num*i+num]=int_f/2
    if i!=0 and i!=n_cell-1:
        Average2[num*i:num*i+num,num*i-1:num*i+num+1]=block1
    for j in range(num):
        Average1[num*i+j,num*i:num*i+num]=int_f/2
    if i<n_cell-1:
        Fg[num*i+num-1,num*i+num-1],Fg[num*i+num,num*i+num-1],\
        Fg[num*i+num-1,num*i+num],Fg[num*i+num,num*i+num]=\
        -1/2,1/2,-1/2,1/2
        Fg1[num*i+num-1,num*i+num-1],Fg1[num*i+num,num*i+num-1],\
        Fg1[num*i+num-1,num*i+num],Fg1[num*i+num,num*i+num]=\
        -1/2,1/2,1/2,-1/2
        Fu[num*i+num-1,num*i+num-1],Fu[num*i+num,num*i+num-1],\
        Fu[num*i+num-1,num*i+num],Fu[num*i+num,num*i+num]=\
        0,1,1,0
inv_MK=np.linalg.inv(MK)
#转化为torch使用的张量
x=torch.from_numpy(x).float()
U=torch.from_numpy(U).float()
MK=torch.from_numpy(MK).float()
SK=torch.from_numpy(SK).float()
Fg=torch.from_numpy(Fg).float()
Fg1=torch.from_numpy(Fg1).float()
Fu=torch.from_numpy(Fu).float()
inv_MK=torch.from_numpy(inv_MK).float()
Average=torch.from_numpy(Average).float()
Average1=torch.from_numpy(Average1).float()
Average2=Average1+Average2@Average1#一阶平均逼近

d1=torch.zeros(num*n_cell,num*n_cell)
d2=torch.zeros(num*n_cell,num*n_cell)
d3=torch.zeros(num*n_cell,num*n_cell)
d4=torch.zeros(num*n_cell,num*n_cell)
for i in range(num*n_cell):
    if i>num-1:
        d1[i,[i-num,i]]=torch.tensor([-1.,1.])
    if i<num*n_cell-num:
        d2[i,[i,i+num]]=torch.tensor([-1.,1.])
    d3[i,i//num*num]=torch.tensor(1.)
    d4[i,i//num*num+num-1]=torch.tensor(1.)
d1[:num,:]=d2[:num,:]
d2[num*n_cell-num:,:]=d1[num*n_cell-num:,:]
d1=d1@Average1
d2=d2@Average1
d3=Average1-d3
d4=d4-Average1

###############################定义神经网络与函数################################

def minmod(A):#最小模限制，有正有负就取0
    l=A.shape[0]
    s=torch.sign(A).sum(0,keepdims=True)/l
    mm=torch.abs(A).min(0,keepdims=True)[0] * torch.floor(torch.abs(s))*s
    return mm
def minmod1(A):#最小模限制，取最接近于0的数
    s=torch.abs(A).min(0,keepdims=True)[0]
    mm=[A[torch.where(torch.abs(A[:,i])==s[0,i]),i] for i in range(4)]
    mm=torch.cat([mm[i][:,0:1] for i in range(4)],1)
    return mm
def Lambd1(U):#Lax-Friedrich通量
    U1=U
    h1,u1,h2,u2 = U1[:,0:1],U1[:,1:2],U1[:,2:3],U1[:,3:4]
    lambd=torch.abs(torch.cat([(h1*u1+h2*u2)/(h1+h2)+(g*(h1+h2))**0.5,
                      (h1*u1+h2*u2)/(h1+h2)-(g*(h1+h2))**0.5,
          (h1*u2+h2*u1)/(h1+h2)+((1-r)*g*(h1*h2)/(h1+h2)*\
            (1-(u2-u1)**2/(1-r)/g/(h1+h2)))**0.5,
          (h1*u2+h2*u1)/(h1+h2)-((1-r)*g*(h1*h2)/(h1+h2)*\
            (1-(u2-u1)**2/(1-r)/g/(h1+h2)))**0.5],1))
    lambd=torch.cat([Fu@lambd,lambd],1).max(1,keepdims=True)[0]
    # max_lambd=lambd.max()
    lambd1=Fg1*lambd
    return lambd1@U
# U_i=torch.zeros(num*n_cell,Nc)
# Fu1=torch.zeros(num*n_cell,num*n_cell)
# Fu2=torch.zeros(num*n_cell,num*n_cell)
# Fu3=torch.zeros(num*n_cell,num*n_cell)
# Fu4=torch.zeros(num*n_cell,num*n_cell)
# for i in range(n_cell-1):
#     U_i[num*i+num-1,:]=torch.tensor([1.0]*Nc)
#     U_i[num*i+num,:]=torch.tensor([1.0]*Nc)
#     Fu1[num*i+num-1,num*i+num-2],Fu1[num*i+num,num*i+num-2]=1,1
#     Fu2[num*i+num-1,num*i+num-1],Fu2[num*i+num,num*i+num-1]=1,1
#     Fu3[num*i+num-1,num*i+num],Fu3[num*i+num,num*i+num]=1,1
#     Fu4[num*i+num-1,num*i+num+1],Fu4[num*i+num,num*i+num+1]=1,1
# def Lambd2(U):#神经网络定义的通量
#     U3=torch.cat([Fu2@U,Fu3@U],-1)
#     lambd=U_i*NN1(U3)
#     return lambd

##########define update function
M=3#虚拟步长的倍数
D1=torch.zeros(num*n_cell,num*n_cell)
D2=torch.zeros(num*n_cell,num*n_cell)
D=torch.zeros(num*n_cell,num*n_cell)
for i in range(1,num*n_cell-1):
    D[i,i:i+2]=torch.tensor([-1,1])
    if i%num==0:
        D1[i,i-1:i+2]=torch.tensor([-M**2/(M+1),M-1,1/(M+1)])
        D2[i,i-1:i+2]=torch.tensor([2*M**2/(M+1),-2*M,2*M/(M+1)])
    elif i%num==num-1:
        D[i,i:i+2]=D[i,i:i+2]*M
        D1[i,i-1:i+2]=torch.tensor([-1/(M+1),-(M-1),M**2/(M+1)])
        D2[i,i-1:i+2]=torch.tensor([2*M/(M+1),-2*M,2*M**2/(M+1)])
    else:
        D1[i,i-1:i+2]=torch.tensor([-1/2,0,1/2])
        D2[i,i-1:i+2]=torch.tensor([1,-2,1])
D1[0]=D1[2];D1[-1]=D1[-3]
D2[0,:]=D2[2,:];D2[-1]=D2[-3,:]
D[0,:2]=torch.tensor([-1,1]);D[-1,-2:]=torch.tensor([-1,1])

M1=6
D_1=torch.zeros(num*n_cell,num*n_cell)
D_2=torch.zeros(num*n_cell,num*n_cell)
for i in range(1,num*n_cell-1):
    if i%num==0:
        D_1[i,i-1:i+2]=torch.tensor([-M**2/(M+1),M-1,1/(M+1)])
        D_2[i,i-1:i+2]=torch.tensor([2*M1**2/(M1+1),-2*M1,2*M1/(M1+1)])
    elif i%num==2:
        D_1[i,i-1:i+2]=torch.tensor([-1/(M+1),-(M-1),M**2/(M+1)])
        D_2[i,i-1:i+2]=torch.tensor([2*M1/(M1+1),-2*M1,2*M1**2/(M1+1)])
    else:
        D_1[i,i-1:i+2]=torch.tensor([-1/2,0,1/2])
        D_2[i,i-1:i+2]=torch.tensor([1,-2,1])
D_1[0]=D_1[2];D_1[-1]=D_1[-3]
D_2[0,:]=D_2[2,:];D_2[-1]=D_2[-3,:]

num4=3
nodes1=gLLNodesAndWeights(num4)[0]
trans=np.concatenate([LagP(i,nodes1,nodes)[:,None] for i in range(num4)],1)
evl=torch.zeros(num*n_cell,num4*n_cell)
for i in range(n_cell):
    evl[i*num:i*num+num,num4*i:num4*i+num4]=torch.from_numpy(trans)

# W=torch.zeros(num*n_cell,num*n_cell)
# w1=torch.tensor([1,0,0,1,0,0,1,0,0,1,0,0,1])
# for i in range(6,num*n_cell-6):
#     W[i,i-6:i+7]=w1
# W[0,:7]=w1[6:];W[1,:8]=w1[5:];W[2,:9]=w1[4:]
# W[3,:10]=w1[3:];W[4,:11]=w1[2:];W[5,:12]=w1[1:]
# W[-1,-7:]=w1[:-6];W[-2,-8:]=w1[:-5];W[-3,-9:]=w1[:-4];
# W[-4,-10:]=w1[:-3];W[-5,-11:]=w1[:-2];W[-6,-12:]=w1[:-1]
def limiter(U2):
    U3=Average@U2
    for i in range(0,n_cell):#斜率限制器
        a=minmod(torch.cat([U3[i:i+1]-U2[num*i:num*i+1],\
            U3[i:i+1]-U3[i-1:i],U3[i+1:i+2]-U3[i:i+1]],0))
        b=minmod(torch.cat([U2[num*i+num-1:num*i+num]\
            -U3[i:i+1],U3[i:i+1]-U3[i-1:i],U3[i+1:i+2]-U3[i:i+1]],0))
        if (torch.abs(U3[i]-U2[num*i]-a)+torch.abs(U2[num*i+num-1]-U3[i]-b)).sum()>1e-6:
            for j in range(num):
                c=minmod(torch.cat([(U3[i:i+1]-U3[i-1:i])/dx,\
                                          (U3[i+1:i+2]-U3[i:i+1])/dx,\
                                          (U3[i+1:i+2]-U3[i-1:i])/dx/2],0))
                U2[num*i+j]=U3[i]+(x[num*i+j]-x[num*i]-dx/2)*c
    return U2
def update(U,dt,func):#func是通量函数
    # K1=inv_MK@((Fg+SK)@G(U,x) + func(U)) #使用Friedrich-Lax通量,TVD-RK
    # K2=inv_MK@((Fg+SK)@G(U+K1*dt,x) + func(U+K1*dt))
    # K3=inv_MK@((Fg+SK)@G(U+(K1+K2)*1/4*dt,x) + func(U+(K1+K2)*1/4*dt))
    # U2=U+dt*(K1*1/6+K2*1/6+K3*2/3)
    
    U11=U + inv_MK@((Fg+SK)@G(U,x) + func(U))*dt
    # U11[[0,-1,0,-1],[1,1,3,3]]=0
    # U11[:num,1]=U11[-num:,1]=U11[:num,3]=U11[-num:,3]=0
    # U11=NN2(U11.t().reshape(-1,1,n_cell*num)).detach().reshape(-1,n_cell*num).t()
    # U11=limiter(U11)
    inde=torch.where(U11[:,[0,2]]<0)
    U11[inde[0],inde[1]*2]=1e-8
    U12=U*3/4 + U11*1/4 + inv_MK@((Fg+SK)@G(U11,x) + func(U11))*dt*1/4
    # U12[[0,-1,0,-1],[1,1,3,3]]=0
    # U12[:num,1]=U12[-num:,1]=U12[:num,3]=U12[-num:,3]=0
    # U12=NN2(U12.t().reshape(-1,1,n_cell*num)).detach().reshape(-1,n_cell*num).t()
    # U12=limiter(U12)
    inde=torch.where(U12[:,[0,2]]<0)
    U12[inde[0],inde[1]*2]=1e-8
    U2=U*1/3 + U12*2/3 + inv_MK@((Fg+SK)@G(U12,x) + func(U12))*dt*2/3
    U2[[0,-1,0,-1],[1,1,3,3]]=0
    # U2[:num,1]=U2[-num:,1]=U2[:num,3]=U2[-num:,3]=0
    U2=NN2(U2.t().reshape(-1,1,n_cell*num)).detach().reshape(-1,n_cell*num).t()
    # U2=limiter(U2)
    inde=torch.where(U2[:,[0,2]]<0)
    U2[inde[0],inde[1]*2]=1e-8
    
    return U2

#%% generate data
from torch import sin
num3=2000
mul=2*torch.pi/(x_ran[1]-x_ran[0])

DATA=[]
DATA1=[]
DD=[]
Index1=[]
for i in range(num3):
    index1=0
    y1=torch.zeros(num*n_cell,4)
    y11=torch.zeros(num*n_cell,4)
    ind1=torch.zeros(num*n_cell,4)
    while index1<num*n_cell-50:
        if index1!=0:
            ind1[index1,:]=1
        a=torch.rand(8,4)*2
        y1[index1:]=a[0,:]+a[1,:]*sin(x[index1:]*mul)\
            +a[2,:]*sin(x[index1:]*2*mul)+\
            a[3,:]*sin(x[index1:]*3*mul)+\
            a[4,:]*sin(x[index1:]*4*mul)
            # a[5,:]*sin(x[index1:]*5*mul)+\
            # a[6,:]*sin(x[index1:]*6*mul)
        y11[index1:]=2*a[0,:]*torch.ones_like(x[index1:])
        index1=index1+torch.randint(50,65,(1,))
    DATA1.append(y1)
    DATA1.append(-y1)
    DATA1.append(y1[range(num*n_cell-1,-1,-1)])
    DATA1.append(y11)
    Index1.append(ind1)
    s1=(torch.einsum('lk,ki->li',D2,y1).pow(2)).sum(0).mean()
    DD.append(s1)
    ind2=ind1.clone()
    for j in range(1,10):
        ind2[j:,:]=ind2[j:,:]+ind1[:-j,:]
        ind2[:-j,:]=ind2[:-j,:]+ind1[j:,:]
    eps=ind2*torch.randn(num*n_cell,4)*1
    y2=y1 + (eps-Average1@eps)
    y21=y11 + (eps-Average1@eps)
    s2=(torch.einsum('lk,ki->li',D2,y2).pow(2)).sum(0).mean()
    # if i%100==0:
    #     print('s1: ',s1)
    #     print(s2-s1)
    DATA.append(y2)
    DATA.append(-y2)
    DATA.append(y2[range(num*n_cell-1,-1,-1)])
    DATA.append(y21)

#%% DNN训练
num1=1;num2=101

# DATA=np.load('DATA1.npy',allow_pickle=True)
DATA=torch.cat([DATA[i][None,:,:] for i in range(len(DATA))],0)
V0=DATA[:].permute([2,0,1]).reshape(-1,1,num*n_cell)
# V0_mean=V0.mean(dim=-1,keepdims=True)
# V0_var=V0.var(dim=-1,keepdims=True)
V1=torch.cat([DATA1[i][None,:,:] for i in range(len(DATA1))],0).\
    permute([2,0,1]).reshape(-1,1,num*n_cell)
# Index=torch.cat([Index1[i][None,:,:] for i in range(len(Index1))],0).\
#     permute([2,0,1]).reshape(-1,1,num*n_cell)
# dV=torch.zeros_like(V0)
# dV[:,:,:-1]=V0[:,:,1:]-V0[:,:,:-1];dV[:,:,-1]=dV[:,:,-2]
# ddV=torch.einsum('lk,ijk->ijl',D2,V0)
# # A=torch.eye(num*n_cell)-Average1
# ind=torch.where(dV.abs()>dV.abs().mean(2,keepdims=True)*6)
# dV_=torch.clone(dV)
# dV_[ind[0],ind[1],ind[2]]=0
# V0_=torch.clone(V0)
# for i in range(1,num*n_cell):
#     V0_[:,:,i]=V0_[:,:,0]+dV_[:,:,:i].sum(2)

torch.manual_seed(1)
# NN1=PDENN([Nc*2]+[20,20,20,20]+[Nc])
NN2=PDECNN1(30)
NN2.check=0

# optim1=torch.optim.Adam(NN1.parameters(), 1e-3, betas = [0.9, 0.999])
optim2=torch.optim.Adam(NN2.parameters(), 1e-3, betas = [0.9, 0.999], weight_decay=0)
for group in optim2.param_groups:
    group.setdefault('initial_lr', group['lr'])
sched2 = torch.optim.lr_scheduler.MultiStepLR(optim2, milestones=\
        [10000,12000,14000,16000,20000], gamma=0.5, last_epoch=10000)

step=0
t=time()
Loss=[]
fig = plt.figure(num=1, figsize=(16, 8),dpi=150)
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)
while step<=50000:
    step+=1
    j=step%(len(V0)//100)
    Vo=V0[100*j:100*j+100]
    V=NN2(Vo)
    
    loss1=torch.einsum('lk,ijk->ijl',D_2,V).pow(2)
    loss2=loss1[:,:,[i for i in range(40)]+[i-40 for i in range(40)]].sum(2).mean()
    # loss3=loss1[:,:,1::3].sum(2).mean()
    # Vp=NN2.padding(V,50)
    # xp=torch.cat([2*x[0:1]-x[range(50,0,-1)],x,2*x[-1:]-x[range(-2,-2-50,-1)]],0)
    x1=x/dx*deg*0.05 #注意空间间距不能过低
    loss4=torch.cat([(V[:,:,1:]-V[:,:,:-1])[None,:,:,:],\
            (x1[1:,0]-x1[:-1,0]).repeat(1,V.shape[0],V.shape[1],1)],0).\
        norm(dim=0).sum(-1).mean()
    # loss5=torch.einsum('lk,ijk->ijl',D_1,V).pow(2).sum(2).mean()
    # loss5=(V1[100*j:100*j+100]-V).pow(2).mean()
    loss=loss4+loss2
    
    loss.backward(retain_graph=True)
    # optim1.step()
    optim2.step()
    sched2.step()
    # optim1.zero_grad()
    optim2.zero_grad()
    
    Loss.append(loss.detach())
    loss_av=torch.tensor(Loss[-num2:]).mean()
    
    if j == 1:
        ax1.cla()
        ax1.set_xlim(x_ran)
        ax1.set_ylim([-10,10])
        ax1.plot(x,V[:4,:,:].detach().reshape(-1,num*n_cell).t())
        ax2.cla()
        ax2.set_xlim(x_ran)
        ax2.set_ylim([-10,10])
        ax2.plot(x,V1[j*100:j*100+4,:,:].detach().reshape(-1,num*n_cell).t(),\
                  x,V[:4,:,:].detach().reshape(-1,num*n_cell).t())
        plt.pause(0.01)
        plt.show()
        print("训练: step {}, loss {:.4e}, train_time {:.4e}".
              format(step, loss.item(), time()-t))

# torch.save(NN2,'./Model13')

# NN2=torch.load('./Model13')
# NN3=PDECNN3(30)
# NN3.load_state_dict(NN2.state_dict())

# plt.plot(x,DATA1[113][:,1])

#%% Main loop and plot
t = 0.0
# dt=dx/40
dt=1/800/40
dx1=(x_ran[1]-x_ran[0])/400
x_s1=np.linspace(x_ran[0],x_ran[1],400+1)
x1=np.concatenate([nodes.reshape(-1,1)*dx1/deg+i-dx1/deg for i in x_s1[1:]],0)
x1.sort(0)

UU=[U]
i=0
# NN2.check=1
t1=time()
while t <= t_ran[1]:
    U = update(U,dt,Lambd1).detach()
    UU.append(U)
    t += dt
    if i%100==0:
        print('t:{}'.format(t))
    i+=1
print("{}s".format(time()-t1))

UU_real=torch.load('./UU_real2')

fig = plt.figure(num=1, figsize=(16, 8), dpi=150)
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
t = 0.0
j = 3200
# for j in range(len(UU)):
# Plot the results
U = UU[j]
Ur=UU_real[j]
h1, u1, h2, u2 = U[:, 0:1].detach(), U[:, 1:2].detach(),\
    U[:, 2:3].detach(), U[:, 3:4].detach()
h11, u11, h21, u21 = Ur[:, 0:1].detach(), Ur[:, 1:2].detach(),\
    Ur[:, 2:3].detach(), Ur[:, 3:4].detach()
ax1.cla()
ax1.set_xlim(x_ran)
ax1.set_xlabel('x')
ax1.set_ylim([-1, 4])
ax1.set_ylabel('depth')
ax1.plot(x, h1+h2, 'g', label='h1+h2', linewidth=1)
ax1.plot(x, h2, 'b', label='h2', linewidth=1)
ax1.plot(x1, h11+h21, 'g--', label='real h1+h2', linewidth=1)
ax1.plot(x1, h21, 'b--', label='real h2', linewidth=1)
ax1.plot(x, B(x), 'k', label='bottom')
ax1.legend()
ax2.cla()
ax2.set_xlim(x_ran)
ax2.set_xlabel('x')
ax2.set_ylim([-3.1, 3.1])
ax2.set_ylabel('velocity')
ax2.plot(x, u1, 'g', label='u1', linewidth=1)
ax2.plot(x, u2, 'b', label='u2', linewidth=1)
ax2.plot(x1, u11, 'g--', label='real u1', linewidth=1)
ax2.plot(x1, u21, 'b--', label='real u2', linewidth=1)

ax2.legend()
t=dt*j
plt.savefig('./image/figSW{:.2f}.png'.format(t))
plt.show()

#%% 
fig = plt.figure(num=1, figsize=(16, 8),dpi=150)
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)
t = 0.0
for j in range(len(UU)):
    # Plot the results
    U=UU[j]
    h1,u1,h2,u2 = U[:,0:1].detach(),U[:,1:2].detach(),\
        U[:,2:3].detach(),U[:,3:4].detach()
    ax1.cla()
    ax1.set_xlim(x_ran)
    ax1.set_xlabel('x')
    ax1.set_ylim([-1,4])
    ax1.set_ylabel('depth')
    ax1.plot(x,h1 + h2,'g',label='h1')
    ax1.plot(x,h2,'b',label='h2')
    ax1.plot(x,B(x),'k',label='bottom')
    ax1.legend()
    ax2.cla()
    ax2.set_xlim(x_ran)
    ax2.set_xlabel('x')
    ax2.set_ylim([-3.1,3.1])
    ax2.set_ylabel('velocity')
    ax2.plot(x,u1,'g',label='u1')
    ax2.plot(x,u2,'b',label='u2')
    ax2.legend()
    if abs(t-0)<=dt/2 or abs(t-0.5)<=dt/2 or abs(t-0.7)<=dt/2 or abs(t-1)<=dt/2 \
        or abs(t-1.5)<=dt or abs(t-1.7)<=dt:
        plt.savefig('./image/figSW{:.2f}.png'.format(t))
    t += dt
    plt.pause(dt*0.01)
    plt.show()

# torch.save(UU,'./UU_{}'.format(n_cell))
# DATA=(DATA+UU)[-3000:][::-1]



