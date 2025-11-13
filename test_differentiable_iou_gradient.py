"""
Differentiable Polygon IoU Gradient 흐름 테스트

이 스크립트는 새로 구현한 differentiable_quad_iou_8coords 함수가
올바르게 gradient를 전파하는지 확인합니다.
"""

import torch
from ultralytics.utils.metrics import differentiable_quad_iou_8coords, quad_iou_8coords


def test_gradient_flow():
    """Gradient가 올바르게 흐르는지 테스트"""
    print("="*80)
    print("Differentiable Polygon IoU Gradient Test")
    print("="*80 + "\n")

    # 테스트 데이터 생성 (배치 크기 4)
    batch_size = 4

    # 사각형 1: requires_grad=True로 설정
    quad1 = torch.tensor([
        [100.0, 100.0, 200.0, 100.0, 200.0, 200.0, 100.0, 200.0],  # 정사각형
        [150.0, 150.0, 250.0, 150.0, 250.0, 250.0, 150.0, 250.0],  # 정사각형 (겹침)
        [50.0, 50.0, 150.0, 60.0, 140.0, 150.0, 40.0, 140.0],      # 약간 비틀린 사각형
        [300.0, 300.0, 400.0, 300.0, 400.0, 400.0, 300.0, 400.0],  # 멀리 떨어진 사각형
    ], requires_grad=True)

    # 사각형 2: 고정
    quad2 = torch.tensor([
        [150.0, 150.0, 250.0, 150.0, 250.0, 250.0, 150.0, 250.0],  # 정사각형 (겹침)
        [100.0, 100.0, 200.0, 100.0, 200.0, 200.0, 100.0, 200.0],  # 정사각형
        [80.0, 80.0, 180.0, 85.0, 175.0, 180.0, 75.0, 175.0],      # 약간 비틀린 사각형 (겹침)
        [100.0, 100.0, 200.0, 100.0, 200.0, 200.0, 100.0, 200.0],  # 겹치지 않음
    ])

    print("1. Differentiable Polygon IoU 테스트")
    print("-" * 80)

    # Forward pass
    iou_new = differentiable_quad_iou_8coords(quad1, quad2)
    print(f"IoU 값: {iou_new.squeeze().detach()}")

    # Loss 계산 (1 - IoU)
    loss = (1.0 - iou_new).mean()
    print(f"Loss (1 - IoU): {loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Gradient 확인
    if quad1.grad is not None:
        print(f"\n✅ Gradient 흐름 확인:")
        print(f"   - quad1.grad.shape: {quad1.grad.shape}")
        print(f"   - quad1.grad 통계:")
        print(f"     * mean: {quad1.grad.mean().item():.6f}")
        print(f"     * std:  {quad1.grad.std().item():.6f}")
        print(f"     * min:  {quad1.grad.min().item():.6f}")
        print(f"     * max:  {quad1.grad.max().item():.6f}")
        print(f"     * norm: {quad1.grad.norm().item():.6f}")

        # Gradient가 0이 아닌지 확인
        non_zero_grad = (quad1.grad.abs() > 1e-8).sum().item()
        total_grad = quad1.grad.numel()
        print(f"   - Non-zero gradients: {non_zero_grad}/{total_grad} ({100*non_zero_grad/total_grad:.1f}%)")

        if non_zero_grad > 0:
            print("\n✅ SUCCESS: Gradient가 정상적으로 흐르고 있습니다!")
        else:
            print("\n❌ FAIL: Gradient가 모두 0입니다!")
    else:
        print("\n❌ FAIL: Gradient가 None입니다!")

    print("\n" + "="*80)
    print("2. 기존 AABB Fallback과 비교")
    print("-" * 80)

    # 새로운 quad1 생성 (gradient 초기화)
    quad1_old = quad1.detach().clone().requires_grad_(True)

    # 기존 방식
    iou_old = quad_iou_8coords(quad1_old, quad2, use_shapely=False)
    print(f"AABB IoU 값: {iou_old.squeeze().detach()}")

    loss_old = (1.0 - iou_old).mean()
    print(f"AABB Loss (1 - IoU): {loss_old.item():.4f}")

    loss_old.backward()

    if quad1_old.grad is not None:
        print(f"\nAABB Gradient 통계:")
        print(f"   - mean: {quad1_old.grad.mean().item():.6f}")
        print(f"   - std:  {quad1_old.grad.std().item():.6f}")
        print(f"   - norm: {quad1_old.grad.norm().item():.6f}")

    print("\n" + "="*80)
    print("3. IoU 값 비교 (정확도)")
    print("-" * 80)

    iou_diff = (iou_new - iou_old).abs()
    print(f"New IoU: {iou_new.squeeze().detach()}")
    print(f"Old IoU: {iou_old.squeeze().detach()}")
    print(f"차이:     {iou_diff.squeeze().detach()}")
    print(f"평균 차이: {iou_diff.mean().item():.6f}")

    print("\n" + "="*80)
    print("테스트 완료!")
    print("="*80)


def test_gradient_numerical():
    """Numerical gradient와 비교하여 정확성 검증"""
    print("\n" + "="*80)
    print("Numerical Gradient 검증")
    print("="*80 + "\n")

    # 간단한 케이스로 테스트
    quad1 = torch.tensor([[100.0, 100.0, 200.0, 100.0, 200.0, 200.0, 100.0, 200.0]], requires_grad=True)
    quad2 = torch.tensor([[150.0, 150.0, 250.0, 150.0, 250.0, 250.0, 150.0, 250.0]])

    # Analytical gradient
    iou = differentiable_quad_iou_8coords(quad1, quad2)
    loss = (1.0 - iou).sum()
    loss.backward()
    analytical_grad = quad1.grad.clone()

    # Numerical gradient (finite difference)
    epsilon = 1e-4
    numerical_grad = torch.zeros_like(quad1)

    for i in range(quad1.shape[1]):
        # f(x + epsilon)
        quad1_plus = quad1.detach().clone()
        quad1_plus[0, i] += epsilon
        iou_plus = differentiable_quad_iou_8coords(quad1_plus, quad2)
        loss_plus = (1.0 - iou_plus).sum()

        # f(x - epsilon)
        quad1_minus = quad1.detach().clone()
        quad1_minus[0, i] -= epsilon
        iou_minus = differentiable_quad_iou_8coords(quad1_minus, quad2)
        loss_minus = (1.0 - iou_minus).sum()

        # Numerical gradient: (f(x+eps) - f(x-eps)) / (2*eps)
        numerical_grad[0, i] = (loss_plus - loss_minus) / (2 * epsilon)

    print(f"Analytical gradient: {analytical_grad}")
    print(f"Numerical gradient:  {numerical_grad}")
    print(f"차이: {(analytical_grad - numerical_grad).abs()}")
    print(f"평균 상대 오차: {((analytical_grad - numerical_grad).abs() / (numerical_grad.abs() + 1e-8)).mean().item():.6f}")

    relative_error = ((analytical_grad - numerical_grad).abs() / (numerical_grad.abs() + 1e-8)).mean().item()

    if relative_error < 0.01:  # 1% 이내 오차
        print("\n✅ SUCCESS: Gradient 계산이 정확합니다!")
    else:
        print(f"\n⚠️  WARNING: Gradient 오차가 큽니다 ({relative_error*100:.2f}%)")


if __name__ == "__main__":
    # 기본 gradient 흐름 테스트
    test_gradient_flow()

    # Numerical gradient 검증
    test_gradient_numerical()
