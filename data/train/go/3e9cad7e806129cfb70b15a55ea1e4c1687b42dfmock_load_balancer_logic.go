// Automatically generated by MockGen. DO NOT EDIT!
// Source: github.com/quintilesims/layer0/api/logic (interfaces: LoadBalancerLogic)

package mock_logic

import (
	gomock "github.com/golang/mock/gomock"
	models "github.com/quintilesims/layer0/common/models"
)

// Mock of LoadBalancerLogic interface
type MockLoadBalancerLogic struct {
	ctrl     *gomock.Controller
	recorder *_MockLoadBalancerLogicRecorder
}

// Recorder for MockLoadBalancerLogic (not exported)
type _MockLoadBalancerLogicRecorder struct {
	mock *MockLoadBalancerLogic
}

func NewMockLoadBalancerLogic(ctrl *gomock.Controller) *MockLoadBalancerLogic {
	mock := &MockLoadBalancerLogic{ctrl: ctrl}
	mock.recorder = &_MockLoadBalancerLogicRecorder{mock}
	return mock
}

func (_m *MockLoadBalancerLogic) EXPECT() *_MockLoadBalancerLogicRecorder {
	return _m.recorder
}

func (_m *MockLoadBalancerLogic) CreateLoadBalancer(_param0 models.CreateLoadBalancerRequest) (*models.LoadBalancer, error) {
	ret := _m.ctrl.Call(_m, "CreateLoadBalancer", _param0)
	ret0, _ := ret[0].(*models.LoadBalancer)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

func (_mr *_MockLoadBalancerLogicRecorder) CreateLoadBalancer(arg0 interface{}) *gomock.Call {
	return _mr.mock.ctrl.RecordCall(_mr.mock, "CreateLoadBalancer", arg0)
}

func (_m *MockLoadBalancerLogic) DeleteLoadBalancer(_param0 string) error {
	ret := _m.ctrl.Call(_m, "DeleteLoadBalancer", _param0)
	ret0, _ := ret[0].(error)
	return ret0
}

func (_mr *_MockLoadBalancerLogicRecorder) DeleteLoadBalancer(arg0 interface{}) *gomock.Call {
	return _mr.mock.ctrl.RecordCall(_mr.mock, "DeleteLoadBalancer", arg0)
}

func (_m *MockLoadBalancerLogic) GetLoadBalancer(_param0 string) (*models.LoadBalancer, error) {
	ret := _m.ctrl.Call(_m, "GetLoadBalancer", _param0)
	ret0, _ := ret[0].(*models.LoadBalancer)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

func (_mr *_MockLoadBalancerLogicRecorder) GetLoadBalancer(arg0 interface{}) *gomock.Call {
	return _mr.mock.ctrl.RecordCall(_mr.mock, "GetLoadBalancer", arg0)
}

func (_m *MockLoadBalancerLogic) ListLoadBalancers() ([]*models.LoadBalancerSummary, error) {
	ret := _m.ctrl.Call(_m, "ListLoadBalancers")
	ret0, _ := ret[0].([]*models.LoadBalancerSummary)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

func (_mr *_MockLoadBalancerLogicRecorder) ListLoadBalancers() *gomock.Call {
	return _mr.mock.ctrl.RecordCall(_mr.mock, "ListLoadBalancers")
}

func (_m *MockLoadBalancerLogic) UpdateLoadBalancerHealthCheck(_param0 string, _param1 models.HealthCheck) (*models.LoadBalancer, error) {
	ret := _m.ctrl.Call(_m, "UpdateLoadBalancerHealthCheck", _param0, _param1)
	ret0, _ := ret[0].(*models.LoadBalancer)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

func (_mr *_MockLoadBalancerLogicRecorder) UpdateLoadBalancerHealthCheck(arg0, arg1 interface{}) *gomock.Call {
	return _mr.mock.ctrl.RecordCall(_mr.mock, "UpdateLoadBalancerHealthCheck", arg0, arg1)
}

func (_m *MockLoadBalancerLogic) UpdateLoadBalancerPorts(_param0 string, _param1 []models.Port) (*models.LoadBalancer, error) {
	ret := _m.ctrl.Call(_m, "UpdateLoadBalancerPorts", _param0, _param1)
	ret0, _ := ret[0].(*models.LoadBalancer)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

func (_mr *_MockLoadBalancerLogicRecorder) UpdateLoadBalancerPorts(arg0, arg1 interface{}) *gomock.Call {
	return _mr.mock.ctrl.RecordCall(_mr.mock, "UpdateLoadBalancerPorts", arg0, arg1)
}
